import glob, os
import gmsh
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dolfinx import fem
from pathlib import Path
from tools.airflow_estimator import AirflowEstimator
from tools.visualizer import Visualizer


def velocity_rmse(u_true: fem.Function, u_est: fem.Function) -> float:
    true_arr = u_true.x.array.reshape(-1, u_true.function_space.dofmap.bs)
    est_arr = u_est.x.array.reshape(-1, u_est.function_space.dofmap.bs)
    diff = est_arr - true_arr
    return float(np.sqrt(np.mean(np.sum(diff**2, axis=1))))

# Example usage 
bpfile   = Path("/app/exp_sample_based_estimation/exp_wind_comparison/data/airflow_10x6_ground_truth.bp")
meshfile = Path("/app/exp_sample_based_estimation/exp_wind_comparison/data/10x6_mesh/mesh.msh")
OUTPUT_DIR = Path("/app/exp_sample_based_estimation/exp_wind_comparison")
GMRF_SUMMARY_CSV = Path("/app/ros2/results/cell0.25/evaluation_summary.csv")
COMPARISON_PLOT = OUTPUT_DIR / "ns_vs_gmrf_rmse_comparison.png"

est = AirflowEstimator.from_bp(bpfile, fun_name="velocity_H2", meshtags_name="facet_tags", p=30, seed=5)
est.set_weights(kin_v=1.5e-5, misfit=1e2, pde_err=1e0, reg=1e-2, boundary=1e5)

# Boundary conditions
V = est.V
Q = est.Q
W0 = est.W0
W1 = est.W1
W = est.W

gmsh.initialize()
gmsh.open(str(meshfile))
phy_groups = gmsh.model.getPhysicalGroups()
name_to_id = {gmsh.model.getPhysicalName(dim, tag): tag for (dim, tag) in phy_groups}
gmsh.finalize()

facet_tags = est.facet_tags
est._boundary_name_to_id = name_to_id

# No slip boundary condition
u_D_no_slip = fem.Function(V)
u_D_no_slip.x.array[:] = 0.0
no_slip_names = ["Walls"]
dofs_local = np.concatenate([facet_tags.find(name_to_id[name]) for name in no_slip_names])
dofs_no_slip = fem.locate_dofs_topological((W0, V), V.mesh.topology.dim - 1, dofs_local)

bc_no_slip = fem.dirichletbc(u_D_no_slip, dofs_no_slip, W0)

# Outflow - zero pressure
p_zero = fem.Function(Q)
p_zero.x.array[:] = 0.0
p_1_facets = ["Outlet"]
dofs_local = np.concatenate([facet_tags.find(name_to_id[name]) for name in p_1_facets])
dofs_out = fem.locate_dofs_topological((W1, Q), V.mesh.topology.dim - 1, dofs_local)

bc_out = fem.dirichletbc(p_zero, dofs_out, W1)

est.add_dirichlet_bc([bc_no_slip, bc_out])

u_true = est.ground_truth.sub(0).collapse()

###### Experiment Loop ######

samples_path = Path("/app/csv_wind_data/10x6_central_obstacle/csv_wind_sample_sets")
os.chdir(samples_path)
sample_files = [samples_path / file for file in glob.glob("*.csv")]

sample_sizes = [10, 25, 50, 100, 200]
summary_rows = []

for n_sample_points in sample_sizes:
    errors = []
    for sample_file in sample_files:
        est.set_measurements_from_csv(sample_file, count=n_sample_points, max_xy_dist=0.2)
        res = est.solve_minimum_residual(25, 1e-2, None, "smooth", False)
        u_est = res.sub(0).collapse()
        errors.append(velocity_rmse(u_true, u_est))

    mean_rmse = float(np.mean(errors))
    std_rmse = float(np.std(errors))
    summary_rows.append(
        {
            "sample_size": n_sample_points,
            "mean_rmse_ns": mean_rmse,
            "std_rmse_ns": std_rmse,
            "num_runs_ns": len(errors),
        }
    )
    print(
        f"n={n_sample_points}: mean_rmse={mean_rmse:.6f}, "
        f"std_rmse={std_rmse:.6f}, num_runs={len(errors)}"
    )

summary = pd.DataFrame(summary_rows).sort_values("sample_size").reset_index(drop=True)
if GMRF_SUMMARY_CSV.exists():
    gmrf_summary = pd.read_csv(GMRF_SUMMARY_CSV).rename(
        columns={
            "mean_rmse": "mean_rmse_gmrf",
            "std_rmse": "std_rmse_gmrf",
            "num_runs": "num_runs_gmrf",
        }
    )
    comparison = summary.merge(gmrf_summary, on="sample_size", how="inner")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(comparison["sample_size"], comparison["mean_rmse_ns"], marker="o", label="NS")
    ax.fill_between(
        comparison["sample_size"],
        comparison["mean_rmse_ns"] - comparison["std_rmse_ns"],
        comparison["mean_rmse_ns"] + comparison["std_rmse_ns"],
        alpha=0.2,
    )
    ax.plot(comparison["sample_size"], comparison["mean_rmse_gmrf"], marker="o", label="GMRF")
    ax.fill_between(
        comparison["sample_size"],
        comparison["mean_rmse_gmrf"] - comparison["std_rmse_gmrf"],
        comparison["mean_rmse_gmrf"] + comparison["std_rmse_gmrf"],
        alpha=0.2,
    )
    ax.set_xlabel("Sample size")
    ax.set_ylabel("Average RMSE (10 random seeds)")
    ax.set_title("Navier-Stokes vs Gaussian Markov Random Field RMSE")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(COMPARISON_PLOT, dpi=160)
    plt.close(fig)

    print("\nNS vs GMRF comparison:")
    print(comparison.to_string(index=False))
    print(f"Saved comparison plot to {COMPARISON_PLOT}")
else:
    print(
        f"\nNo GMRF summary found at {GMRF_SUMMARY_CSV}. "
        "Run ros2/scripts/evaluate_result_files.py first to enable the final comparison."
    )


