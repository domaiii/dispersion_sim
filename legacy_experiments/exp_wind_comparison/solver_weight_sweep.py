from __future__ import annotations

import itertools
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mpi4py import MPI
from dolfinx import fem
from ufl import dx, inner

from tools.airflow_estimator import AirflowEstimator


BPFILE = Path("/app/exp_sample_based_estimation/exp_wind_comparison/airflow_10x6_ground_truth_ipcs.bp")
MESHFILE = Path("/app/exp_sample_based_estimation/exp_wind_comparison/10x6_mesh/mesh.msh")
OUTPUT_DIR = Path("/app/exp_sample_based_estimation/exp_wind_comparison/solver_weight_sweep")

MEASUREMENT_COUNTS = [10, 25, 50, 100, 200]
SEEDS = [0, 1, 2]

KINEMATIC_VISCOSITY = 1.5e-4
REGULARIZATION_MODE = "smooth"


@dataclass(frozen=True)
class SolverSpec:
    method_name: str
    maxit: int
    tol: float
    damping: float | None = None
    uses_boundary_weight: bool = True


SOLVERS: dict[str, SolverSpec] = {
    "minimum_residual": SolverSpec(
        method_name="solve_minimum_residual",
        maxit=25,
        tol=1e-3,
        damping=None,
        uses_boundary_weight=False,
    ),
    "weak_penalty": SolverSpec(
        method_name="solve_weak_penalty",
        maxit=25,
        tol=1e-3,
        damping=None,
        uses_boundary_weight=True,
    ),
    "linear_least_squares": SolverSpec(
        method_name="solve_linear_least_squares",
        maxit=25,
        tol=1e-3,
        damping=None,
        uses_boundary_weight=True,
    ),
}


def build_weight_grid() -> dict[str, list[dict[str, float | None]]]:
    misfit_values = [1.0, 10.0, 100.0]
    pde_values = [0.1, 1.0, 10.0]
    reg_values = [1e-6, 1e-4, 1e-2]
    boundary_values = [10.0, 100.0, 1000.0]

    minimum_residual_grid = []
    for misfit, pde_err, reg in itertools.product(misfit_values, pde_values, reg_values):
        minimum_residual_grid.append(
            {
                "misfit": misfit,
                "pde_err": pde_err,
                "reg": reg,
                "boundary": None,
            }
        )

    penalty_grid = []
    for misfit, pde_err, reg, boundary in itertools.product(
        misfit_values,
        pde_values,
        reg_values,
        boundary_values,
    ):
        penalty_grid.append(
            {
                "misfit": misfit,
                "pde_err": pde_err,
                "reg": reg,
                "boundary": boundary,
            }
        )

    return {
        "minimum_residual": minimum_residual_grid,
        "weak_penalty": penalty_grid,
        "linear_least_squares": penalty_grid,
    }


def l2_norm(expr, domain) -> float:
    local_value = fem.assemble_scalar(fem.form(inner(expr, expr) * dx))
    global_value = domain.comm.allreduce(local_value, op=MPI.SUM)
    return float(np.sqrt(global_value))


def domain_area(domain) -> float:
    local_value = fem.assemble_scalar(fem.form(1.0 * dx(domain=domain)))
    return float(domain.comm.allreduce(local_value, op=MPI.SUM))


def velocity_rmse(u_true: fem.Function, u_est: fem.Function) -> float:
    abs_err = l2_norm(u_est - u_true, u_true.function_space.mesh)
    area = domain_area(u_true.function_space.mesh)
    return abs_err / np.sqrt(area) if area > 1e-14 else 0.0


def directional_rmse(u_true: fem.Function, u_est: fem.Function, eps: float = 1e-12) -> float:
    true_arr = u_true.x.array.reshape(-1, u_true.function_space.dofmap.bs)
    est_arr = u_est.x.array.reshape(-1, u_est.function_space.dofmap.bs)

    true_norm = np.linalg.norm(true_arr, axis=1)
    est_norm = np.linalg.norm(est_arr, axis=1)
    mask = (true_norm > eps) & (est_norm > eps)
    if not np.any(mask):
        return 0.0

    true_dir = true_arr[mask] / true_norm[mask, None]
    est_dir = est_arr[mask] / est_norm[mask, None]
    diff = est_dir - true_dir
    return float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))


def angular_rmse_deg(u_true: fem.Function, u_est: fem.Function, eps: float = 1e-12) -> float:
    true_arr = u_true.x.array.reshape(-1, u_true.function_space.dofmap.bs)
    est_arr = u_est.x.array.reshape(-1, u_est.function_space.dofmap.bs)

    true_norm = np.linalg.norm(true_arr, axis=1)
    est_norm = np.linalg.norm(est_arr, axis=1)
    mask = (true_norm > eps) & (est_norm > eps)
    if not np.any(mask):
        return 0.0

    dots = np.sum(true_arr[mask] * est_arr[mask], axis=1) / (true_norm[mask] * est_norm[mask])
    dots = np.clip(dots, -1.0, 1.0)
    angles_deg = np.degrees(np.arccos(dots))
    return float(np.sqrt(np.mean(angles_deg ** 2)))


def make_estimator(num_measurements: int, seed: int) -> AirflowEstimator:
    est = AirflowEstimator.from_bp(
        BPFILE,
        fun_name="velocity_H2",
        meshtags_name="facet_tags",
        meshfile=MESHFILE,
        p=num_measurements,
        seed=seed,
    )
    est.set_no_slip_bc(["Walls"])
    est.set_zero_pressure_bc(["Outlet"])
    return est


def config_label(config: dict[str, float | None], uses_boundary_weight: bool) -> str:
    base = (
        f"m={config['misfit']:.0e}, "
        f"pde={config['pde_err']:.0e}, "
        f"reg={config['reg']:.0e}"
    )
    if uses_boundary_weight and config["boundary"] is not None:
        return f"{base}, bc={config['boundary']:.0e}"
    return base


def run_single_case(
    est: AirflowEstimator,
    solver_name: str,
    solver_spec: SolverSpec,
    weights: dict[str, float | None],
    num_measurements: int,
    seed: int,
    u_true: fem.Function,
) -> dict[str, float | int | str]:
    est.set_weights(
        kin_v=KINEMATIC_VISCOSITY,
        misfit=float(weights["misfit"]),
        pde_err=float(weights["pde_err"]),
        reg=float(weights["reg"]),
        boundary=weights["boundary"],
    )

    solver = getattr(est, solver_spec.method_name)
    solve_kwargs = {
        "maxit": solver_spec.maxit,
        "tol": solver_spec.tol,
        "regularization": REGULARIZATION_MODE,
        "verbose": False,
    }
    if solver_spec.damping is not None:
        solve_kwargs["damping"] = solver_spec.damping

    result = solver(**solve_kwargs)

    u_est = result.sub(0).collapse()
    rmse = velocity_rmse(u_true, u_est)
    dir_rmse = directional_rmse(u_true, u_est)
    ang_rmse_deg = angular_rmse_deg(u_true, u_est)

    return {
        "solver": solver_name,
        "num_measurements": num_measurements,
        "seed": seed,
        "misfit": float(weights["misfit"]),
        "pde_err": float(weights["pde_err"]),
        "reg": float(weights["reg"]),
        "boundary": float(weights["boundary"]) if weights["boundary"] is not None else np.nan,
        "rmse": rmse,
        "directional_rmse": dir_rmse,
        "angular_rmse_deg": ang_rmse_deg,
    }


def run_weight_sweep() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    grids = build_weight_grid()
    rows: list[dict[str, float | int | str]] = []

    total_cases = sum(
        len(grids[solver_name]) * len(MEASUREMENT_COUNTS) * len(SEEDS)
        for solver_name in SOLVERS
    )
    case_idx = 0

    for num_measurements in MEASUREMENT_COUNTS:
        for seed in SEEDS:
            est = make_estimator(num_measurements=num_measurements, seed=seed)
            u_true = est.ground_truth.sub(0).collapse()

            for solver_name, solver_spec in SOLVERS.items():
                for config_idx, weights in enumerate(grids[solver_name], start=1):
                    label = config_label(weights, solver_spec.uses_boundary_weight)
                    case_idx += 1
                    print(
                        f"[{case_idx:04d}/{total_cases:04d}] "
                        f"{solver_name}, p={num_measurements}, seed={seed}, cfg={config_idx}: {label}"
                    )
                    try:
                        row = run_single_case(
                            est=est,
                            solver_name=solver_name,
                            solver_spec=solver_spec,
                            weights=weights,
                            num_measurements=num_measurements,
                            seed=seed,
                            u_true=u_true,
                        )
                        row["config_id"] = config_idx
                        row["config_label"] = label
                        rows.append(row)
                    except Exception as exc:
                        rows.append(
                            {
                                "solver": solver_name,
                                "num_measurements": num_measurements,
                                "seed": seed,
                                "misfit": float(weights["misfit"]),
                                "pde_err": float(weights["pde_err"]),
                                "reg": float(weights["reg"]),
                                "boundary": float(weights["boundary"]) if weights["boundary"] is not None else np.nan,
                                "rmse": np.nan,
                                "directional_rmse": np.nan,
                                "angular_rmse_deg": np.nan,
                                "config_id": config_idx,
                                "config_label": label,
                                "error": str(exc),
                            }
                        )
                        print(f"  failed: {exc}")

    runs = pd.DataFrame(rows)

    group_cols = [
        "solver",
        "num_measurements",
        "config_id",
        "config_label",
        "misfit",
        "pde_err",
        "reg",
        "boundary",
    ]
    summary = runs.groupby(group_cols, dropna=False).agg(
        mean_rmse=("rmse", "mean"),
        std_rmse=("rmse", "std"),
        mean_directional_rmse=("directional_rmse", "mean"),
        std_directional_rmse=("directional_rmse", "std"),
        mean_angular_rmse_deg=("angular_rmse_deg", "mean"),
        std_angular_rmse_deg=("angular_rmse_deg", "std"),
        count=("rmse", "count"),
    ).reset_index()

    best = (
        summary.sort_values(["solver", "num_measurements", "mean_rmse"])
        .groupby(["solver", "num_measurements"], as_index=False)
        .first()
    )

    return runs, summary, best


def plot_best_error(best: pd.DataFrame, outdir: Path):
    fig, ax = plt.subplots(figsize=(8, 5))
    for solver_name in SOLVERS:
        subset = best[best["solver"] == solver_name].sort_values("num_measurements")
        ax.errorbar(
            subset["num_measurements"],
            subset["mean_rmse"],
            yerr=subset["std_rmse"].fillna(0.0),
            marker="o",
            linewidth=2,
            capsize=4,
            label=solver_name,
        )
    ax.set_xlabel("Number of random measurements")
    ax.set_ylabel("Mean RMSE")
    ax.set_title("Best mean RMSE per solver")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "best_error_vs_measurements.png", dpi=200)
    plt.close(fig)


def plot_best_error_metrics(best: pd.DataFrame, outdir: Path):
    metrics = [
        ("mean_rmse", "std_rmse", "RMSE"),
        ("mean_directional_rmse", "std_directional_rmse", "Directional RMSE"),
        ("mean_angular_rmse_deg", "std_angular_rmse_deg", "Angular RMSE [deg]"),
    ]

    fig, axes = plt.subplots(len(metrics), 1, figsize=(8, 12), sharex=True)
    for ax, (mean_col, std_col, label) in zip(axes, metrics):
        for solver_name in SOLVERS:
            subset = best[best["solver"] == solver_name].sort_values("num_measurements")
            ax.errorbar(
                subset["num_measurements"],
                subset[mean_col],
                yerr=subset[std_col].fillna(0.0),
                marker="o",
                linewidth=2,
                capsize=4,
                label=solver_name,
            )
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
        ax.legend()
    axes[-1].set_xlabel("Number of random measurements")
    fig.suptitle("Best error metrics per solver")
    fig.tight_layout()
    fig.savefig(outdir / "best_error_metrics_vs_measurements.png", dpi=200)
    plt.close(fig)


def plot_best_weights(best: pd.DataFrame, outdir: Path):
    fields = [
        ("misfit", "weight_misfit"),
        ("pde_err", "weight_pde_res"),
        ("reg", "weight_reg"),
        ("boundary", "weight_boundary"),
    ]

    fig, axes = plt.subplots(len(fields), 1, figsize=(8, 12), sharex=True)
    for ax, (column, label) in zip(axes, fields):
        for solver_name, solver_spec in SOLVERS.items():
            if column == "boundary" and not solver_spec.uses_boundary_weight:
                continue
            subset = best[best["solver"] == solver_name].sort_values("num_measurements")
            ax.semilogy(
                subset["num_measurements"],
                subset[column],
                marker="o",
                linewidth=2,
                label=solver_name,
            )
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
        ax.legend()
    axes[-1].set_xlabel("Number of random measurements")
    fig.suptitle("Best weight parameters per solver")
    fig.tight_layout()
    fig.savefig(outdir / "best_weights_vs_measurements.png", dpi=200)
    plt.close(fig)


def plot_heatmaps(summary: pd.DataFrame, outdir: Path):
    fig, axes = plt.subplots(1, len(SOLVERS), figsize=(18, 6), sharey=True)
    if len(SOLVERS) == 1:
        axes = [axes]

    for ax, solver_name in zip(axes, SOLVERS):
        subset = summary[summary["solver"] == solver_name]
        pivot = subset.pivot(
            index="config_id",
            columns="num_measurements",
            values="mean_rmse",
        ).sort_index()

        im = ax.imshow(pivot.to_numpy(), aspect="auto", origin="lower", cmap="viridis")
        ax.set_title(solver_name)
        ax.set_xlabel("Number of random measurements")
        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_xticklabels([str(v) for v in pivot.columns])
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_yticklabels([str(v) for v in pivot.index])
        ax.set_ylabel("Config id")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Mean RMSE")

    fig.suptitle("Weight-grid performance")
    fig.tight_layout()
    fig.savefig(outdir / "solver_error_heatmaps.png", dpi=200)
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    runs, summary, best = run_weight_sweep()

    runs.to_csv(OUTPUT_DIR / "solver_weight_sweep_runs.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "solver_weight_sweep_summary.csv", index=False)
    best.to_csv(OUTPUT_DIR / "solver_weight_sweep_best.csv", index=False)

    plot_best_error(best, OUTPUT_DIR)
    plot_best_error_metrics(best, OUTPUT_DIR)
    plot_best_weights(best, OUTPUT_DIR)
    plot_heatmaps(summary, OUTPUT_DIR)

    print()
    print("Best configurations:")
    print(best.to_string(index=False))
    print()
    print(f"Saved results to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
