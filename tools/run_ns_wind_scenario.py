import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml
from basix.ufl import element
from dolfinx import fem
import dolfinx.io as dio
from mpi4py import MPI
from ufl import dx, inner

from airflow_estimator import AirflowEstimator
from csv_utilities import csv_to_function

SINGLE_LAYER_Z_SPAN = 0.1



@dataclass(frozen=True)
class NsScenarioConfig:
    name: str
    root: Path
    mesh: Path
    wind_csv: Path
    samples_dir: Path
    results_dir: Path
    z_height: float | None = None
    z_tol: float = 0.05
    max_xy_dist: float = 0.2
    wind_sample_sizes: list[int] = field(default_factory=list)
    wall_pattern: str = r"wall|obstacle"
    outflow_pattern: str = r"outlet|outflow"
    solver: str = "minimum_residual"
    regularization: str = "smooth"
    maxit: int = 25
    tol: float = 1e-2
    damping: float | None = None
    viscosity: float = 1e-5
    weight_misfit: float = 1e2
    weight_pde_res: float = 1.0
    weight_reg: float = 1e-2
    weight_boundary: float = 1e4


def load_scenario(path: str | Path) -> NsScenarioConfig:
    scenario_path = Path(path).resolve()
    if scenario_path.is_dir():
        scenario_path = scenario_path / "scenario.yaml"

    with scenario_path.open("r") as f:
        raw = yaml.safe_load(f)

    root = scenario_path.parent
    geometry = raw.get("geometry", {})
    data = raw.get("data", {})
    slicing = raw.get("ground_truth_slicing", {})
    solver = raw.get("ns_solver_parameters", {})
    damping = solver.get("damping", NsScenarioConfig.damping)
    
    return NsScenarioConfig(
        name=str(raw["name"]),
        root=root,
        mesh=(root / geometry["mesh"]).resolve(),
        wind_csv=(root / data["wind_csv"]).resolve(),
        samples_dir=(root / data["sample_dir"]).resolve(),
        results_dir=(root / data["result_dir"]).resolve(),
        z_height=None if slicing.get("z_height") is None else float(slicing["z_height"]),
        z_tol=float(slicing.get("z_tol", NsScenarioConfig.z_tol)),
        max_xy_dist=float(slicing.get("max_xy_dist", NsScenarioConfig.max_xy_dist)),
        wind_sample_sizes=raw.get("wind_sample_sizes", {}),
        wall_pattern=str(geometry.get("wall_pattern", NsScenarioConfig.wall_pattern)),
        outflow_pattern=str(geometry.get("outflow_pattern", NsScenarioConfig.outflow_pattern)),
        solver=str(solver.get("solver", NsScenarioConfig.solver)),
        regularization=str(solver.get("regularization", NsScenarioConfig.regularization)),
        maxit=int(solver.get("maxit", NsScenarioConfig.maxit)),
        tol=float(solver.get("tol", NsScenarioConfig.tol)),
        damping=None if damping is None else float(damping),
        viscosity=float(solver.get("viscosity", NsScenarioConfig.viscosity)),
        weight_misfit=float(solver.get("weight_misfit", NsScenarioConfig.weight_misfit)),
        weight_pde_res=float(solver.get("weight_pde_res", NsScenarioConfig.weight_pde_res)),
        weight_reg=float(solver.get("weight_reg", NsScenarioConfig.weight_reg)),
        weight_boundary=float(solver.get("weight_boundary", NsScenarioConfig.weight_boundary)),
    )


def infer_z_height(wind_csv: Path, z_height: float | None) -> float:
    import pandas as pd

    df = pd.read_csv(wind_csv, usecols=["Points:2"])
    z = df["Points:2"].to_numpy(dtype=float)
    z_min = float(np.min(z))
    z_max = float(np.max(z))

    if z_height is not None:
        return z_height
    if z_max - z_min <= SINGLE_LAYER_Z_SPAN:
        return 0.5 * (z_min + z_max)
    raise ValueError(
        "Ground-truth CSV contains multiple z-levels. Pass a z_height in the scenario config. "
        f"Observed z-range is [{z_min:.6g}, {z_max:.6g}]."
    )


def match_boundary_names(name_to_id: dict[str, int], pattern: str) -> list[str]:
    regex = re.compile(pattern, re.IGNORECASE)
    return [name for name in name_to_id if regex.search(name)]

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
    return float(np.sqrt(np.mean(angles_deg**2)))

def save_velocity_csv(path: Path, velocity: fem.Function) -> None:
    coords = velocity.function_space.tabulate_dof_coordinates()[:, :2]
    values = velocity.x.array.reshape(-1, velocity.function_space.dofmap.bs)
    data = np.column_stack([coords[:, 0], coords[:, 1], values[:, 0], values[:, 1]])
    np.savetxt(path, data, delimiter=",", header="x,y,wind_x,wind_y", comments="")

def solve_estimator(estimator: AirflowEstimator, config: NsScenarioConfig, verbose: bool):
    solver_name = config.solver.strip().lower()
    if solver_name == "minimum_residual":
        return estimator.solve_minimum_residual(
            maxit=config.maxit,
            tol=config.tol,
            damping=config.damping,
            regularization=config.regularization,
            verbose=verbose,
        )
    if solver_name == "weak_penalty":
        return estimator.solve_weak_penalty(
            maxit=config.maxit,
            tol=config.tol,
            damping=config.damping,
            regularization=config.regularization,
            verbose=verbose,
        )
    if solver_name == "linear_least_squares":
        return estimator.solve_linear_least_squares(
            maxit=config.maxit,
            tol=config.tol,
            regularization=config.regularization,
            verbose=verbose,
        )
    raise ValueError(
        f"Unsupported solver {config.solver}, use one of: minimum_residual, weak_penalty, linear_least_squares."
    )

def run_case(config: NsScenarioConfig, sample_csv: Path, sample_size: int | None, verbose: bool) -> dict:
    result_dir = config.results_dir / "ns"
    if sample_size is not None:
        result_dir = result_dir / f"{sample_size}samples"
    result_dir = result_dir / sample_csv.stem
    result_dir.mkdir(parents=True, exist_ok=True)

    domain, _, facet_tags = dio.gmshio.read_from_msh(str(config.mesh), MPI.COMM_WORLD, gdim=2)
    estimator = AirflowEstimator.from_domain(domain, facet_tags, meshfile=config.mesh)

    elem_u = element("Lagrange", domain.basix_cell(), 2, shape=(domain.geometry.dim,))
    V_truth = fem.functionspace(domain, elem_u)
    u_true = fem.Function(V_truth)
    z_height = infer_z_height(config.wind_csv, config.z_height)
    csv_to_function(config.wind_csv, z_height, config.z_tol, u_true, max_xy_dist=config.max_xy_dist)

    w_true = fem.Function(estimator.W)
    w_true.x.array[:] = 0.0
    w_true.sub(0).interpolate(u_true)
    estimator.set_ground_truth(w_true)

    wall_names = match_boundary_names(estimator._boundary_name_to_id, config.wall_pattern)
    if wall_names:
        estimator.set_no_slip_bc(wall_names)

    outflow_names = match_boundary_names(estimator._boundary_name_to_id, config.outflow_pattern)
    if not outflow_names:
        raise ValueError(
            f"No outflow boundaries matched pattern {config.outflow_pattern!r} in {config.mesh.name}."
        )
    estimator.set_zero_pressure_bc(outflow_names)

    estimator.set_regularization(config.regularization)
    estimator.set_weights(
        kin_v=config.viscosity,
        misfit=config.weight_misfit,
        pde_err=config.weight_pde_res,
        reg=config.weight_reg,
        boundary=config.weight_boundary,
    )

    mapping_info = estimator.set_measurements_from_csv(
        sample_csv,
        count=sample_size,
        max_xy_dist=config.max_xy_dist,
    )
    result = solve_estimator(estimator, config, verbose)
    u_est = result.sub(0).collapse()

    metrics = {
        "scenario": config.name,
        "sample_name": sample_csv.stem,
        "sample_size": sample_size if sample_size is not None else int(mapping_info["n_input_samples"]),
        "samples_csv": str(sample_csv),
        "mesh": str(config.mesh),
        "wind_csv": str(config.wind_csv),
        "solver": config.solver,
        "regularization": config.regularization,
        "maxit": config.maxit,
        "tol": config.tol,
        "damping": config.damping,
        "viscosity": config.viscosity,
        "weight_misfit": config.weight_misfit,
        "weight_pde_res": config.weight_pde_res,
        "weight_reg": config.weight_reg,
        "weight_boundary": config.weight_boundary,
        "n_discretization_points": int(u_est.function_space.tabulate_dof_coordinates().shape[0]),
        "rmse": velocity_rmse(u_true, u_est),
        "directional_rmse": directional_rmse(u_true, u_est),
        "angular_rmse_deg": angular_rmse_deg(u_true, u_est),
        **mapping_info,
    }

    estimate_path = result_dir / "wind_estimate_ns.csv"
    metrics_path = result_dir / "metrics_wind_estimate_ns.json"
    save_velocity_csv(estimate_path, u_est)
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    if verbose:
        print(f"Saved estimate to: {estimate_path}")
        print(f"Saved metrics to: {metrics_path}")

    return metrics



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="run_ns_wind",
        description="Run the Navier-Stokes-based wind estimator for one sample CSV or all samples of a scenario.",
    )
    parser.add_argument("scenario", type=str, help="Path to a scenario directory or scenario.yaml.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-s", "--samples", type=str, help="Path to one sample CSV.")
    group.add_argument(
        "--all-samples",
        action="store_true",
        help="Run all sample_points*.csv files in the scenario samples directory.",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args()



def main() -> None:
    args = parse_args()
    config = load_scenario(args.scenario)

    if args.samples:
        sample_files = [Path(args.samples).resolve(strict=True)]
    else:
        sample_files = sorted(config.samples_dir.glob("sample_points*.csv"))
        if not sample_files:
            raise FileNotFoundError(f"No sample_points*.csv files found in {config.samples_dir}")

    sample_sizes = config.wind_sample_sizes or (None,)
    rows = []
    for sample_size in sample_sizes:
        for sample_csv in sample_files:
            if args.verbose:
                label = f" with first {sample_size} samples" if sample_size is not None else ""
                print(f"Running NS wind for {sample_csv.name}{label}")
            rows.append(run_case(config, sample_csv, sample_size, args.verbose))


if __name__ == "__main__":
    main()
