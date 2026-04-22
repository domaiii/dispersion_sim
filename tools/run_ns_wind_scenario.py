import argparse
from contextlib import contextmanager
import json
import os
import re
import time
from pathlib import Path

import numpy as np
from basix.ufl import element
from dolfinx import fem
import dolfinx.io as dio
from mpi4py import MPI
from airflow_estimator import AirflowEstimator
from csv_utilities import csv_to_function
from scenario import ScenarioConfig, infer_z_height


@contextmanager
def suppress_native_output():
    stdout_fd = os.dup(1)
    stderr_fd = os.dup(2)
    try:
        with open(os.devnull, "w", encoding="utf-8") as devnull:
            os.dup2(devnull.fileno(), 1)
            os.dup2(devnull.fileno(), 2)
            yield
    finally:
        os.dup2(stdout_fd, 1)
        os.dup2(stderr_fd, 2)
        os.close(stdout_fd)
        os.close(stderr_fd)



def match_boundary_names(name_to_id: dict[str, int], pattern: str) -> list[str]:
    regex = re.compile(pattern, re.IGNORECASE)
    return [name for name in name_to_id if regex.search(name)]

def save_velocity_csv(path: Path, velocity: fem.Function) -> None:
    coords = velocity.function_space.tabulate_dof_coordinates()[:, :2]
    values = velocity.x.array.reshape(-1, velocity.function_space.dofmap.bs)
    data = np.column_stack([coords[:, 0], coords[:, 1], values[:, 0], values[:, 1]])
    np.savetxt(path, data, delimiter=",", header="x,y,wind_x,wind_y", comments="")

def solve_estimator(estimator: AirflowEstimator, config: ScenarioConfig):
    solver_name = config.solver.strip().lower()
    if solver_name == "minimum_residual":
        return estimator.solve_minimum_residual(
            maxit=config.maxit,
            tol=config.tol,
            damping=config.damping,
            regularization=config.regularization,
            verbose=False,
        )
    if solver_name == "weak_penalty":
        return estimator.solve_weak_penalty(
            maxit=config.maxit,
            tol=config.tol,
            damping=config.damping,
            regularization=config.regularization,
            verbose=False,
        )
    if solver_name == "linear_least_squares":
        return estimator.solve_linear_least_squares(
            maxit=config.maxit,
            tol=config.tol,
            regularization=config.regularization,
            verbose=False,
        )
    raise ValueError(
        f"Unsupported solver {config.solver}, use one of: minimum_residual, weak_penalty, linear_least_squares."
    )

def run_case(config: ScenarioConfig, sample_csv: Path, sample_size: int | None, verbose: bool) -> dict:
    result_dir = config.result_dir / config.solver.strip().lower()
    if sample_size is not None:
        result_dir = result_dir / f"{sample_size}samples"
    result_dir = result_dir / sample_csv.stem
    result_dir.mkdir(parents=True, exist_ok=True)

    with suppress_native_output():
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
        noise_std=config.wind_noise_std,
        max_xy_dist=config.max_xy_dist,
    )
    estimation_start = time.perf_counter()
    result = solve_estimator(estimator, config)
    estimation_runtime_sec = time.perf_counter() - estimation_start
    solver_status = getattr(estimator, "last_solver_status", {})
    status_text = "converged" if solver_status.get("converged") else "reached max iterations"
    iterations = solver_status.get("iterations", "?")
    final_change = solver_status.get("final_relative_change")
    change_text = "nan" if final_change is None else f"{float(final_change):.3e}"
    if verbose:
        print(
            f"NS solver {status_text}"
            f"({sample_size if sample_size is not None else 'all'} samples): "
            f"iterations={iterations}/{config.maxit}, \nfinal_relative_change={change_text},\n tol={config.tol:.3e}"
        )
    u_est = result.sub(0).collapse()

    metadata = {
        "scenario": config.name,
        "estimator": "ns",
        "sample_name": sample_csv.stem,
        "sample_size": sample_size if sample_size is not None else int(mapping_info["n_input_samples"]),
        "samples_csv": str(sample_csv),
        "mesh": str(config.mesh),
        "wind_csv": str(config.wind_csv),
        "wind_noise_std": str(config.wind_noise_std),
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
        "estimation_runtime_sec": float(estimation_runtime_sec),
        "solver_converged": bool(solver_status.get("converged", False)),
        "solver_iterations": int(solver_status.get("iterations", 0)),
        "solver_max_iterations": int(solver_status.get("max_iterations", config.maxit)),
        "solver_final_relative_change": float(solver_status.get("final_relative_change", float("nan"))),
        **mapping_info,
    }

    estimate_path = result_dir / "wind_estimate_ns.csv"
    metadata_path = result_dir / "metadata_wind_est.json"
    save_velocity_csv(estimate_path, u_est)
    metadata["wind_estimate_csv"] = str(estimate_path)
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    if verbose:
        print("---")
        #print(f"Saved estimate to: {estimate_path}")
        #print(f"Saved metadata to: {metadata_path}")

    return metadata



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
    config = ScenarioConfig.load(args.scenario)

    if args.samples:
        sample_files = [Path(args.samples).resolve(strict=True)]
    else:
        sample_files = sorted(config.sample_dir.glob("sample_points*.csv"))
        if not sample_files:
            raise FileNotFoundError(f"No sample_points*.csv files found in {config.sample_dir}")

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
