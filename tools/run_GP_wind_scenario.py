import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tools.GP_wind_estimator import Grid
from scenario import ScenarioConfig


def save_wind_csv(path: Path, grid: Grid, u_field: np.ndarray, v_field: np.ndarray) -> None:
    data = np.column_stack(
        [
            grid.xx[grid.free_mask],
            grid.yy[grid.free_mask],
            u_field[grid.free_mask],
            v_field[grid.free_mask],
        ]
    )
    np.savetxt(path, data, delimiter=",", header="x,y,wind_x,wind_y", comments="")


def infer_scenario_path(input_path: Path) -> Path:
    if input_path.is_file() and input_path.suffix.lower() == ".csv":
        scenario_path = input_path.parent.parent / "scenario.yaml"
        if not scenario_path.exists():
            raise FileNotFoundError(
                f"Could not infer scenario.yaml from sample file {input_path}."
            )
        return scenario_path
    return input_path


def select_sample_files(input_path: Path, config: ScenarioConfig) -> list[Path]:
    if input_path.is_file() and input_path.suffix.lower() == ".csv":
        return [input_path.resolve(strict=True)]

    sample_files = sorted(config.sample_dir.glob("sample_points*.csv"))
    if not sample_files:
        raise FileNotFoundError(f"No sample_points*.csv files found in {config.sample_dir}")
    return sample_files


def run_case(config: ScenarioConfig, sample_csv: Path, sample_size: int | None, verbose: bool) -> dict:
    if config.occupancy_yaml is None:
        raise ValueError("GP wind estimation requires geometry.occupancy_yaml in the scenario config.")

    result_dir = config.result_dir / "gp"
    if sample_size is not None:
        result_dir = result_dir / f"{sample_size}samples"
    result_dir = result_dir / sample_csv.stem
    result_dir.mkdir(parents=True, exist_ok=True)

    grid = Grid.from_occupancy_yaml(config.occupancy_yaml)
    grid.add_csv_measurements(sample_csv, count=sample_size, std_noise=config.wind_noise_std)

    estimation_start = time.perf_counter()
    u_field, v_field = grid.estimate_wind_field_gpr(
        config.gp_length_scale,
        optimize_length_scale=config.gp_optimize_length_scale,
    )
    estimation_runtime_sec = time.perf_counter() - estimation_start

    estimate_path = result_dir / "wind_estimate.csv"
    plot_path = result_dir / "wind_estimate.png"
    metadata_path = result_dir / "metadata_wind_est.json"

    save_wind_csv(estimate_path, grid, u_field, v_field)
    grid.plot_wind_field(u_field, v_field, "GP wind estimate", str(plot_path))

    metadata = {
        "scenario": config.name,
        "estimator": "gp",
        "sample_name": sample_csv.stem,
        "sample_size": sample_size if sample_size is not None else len(grid.measurements),
        "samples_csv": str(sample_csv),
        "wind_csv": str(config.wind_csv),
        "wind_noise_std": str(config.wind_noise_std),
        "occupancy_yaml": str(config.occupancy_yaml),
        "wind_estimate_csv": str(estimate_path),
        "wind_estimate_png": str(plot_path),
        "gp_length_scale": float(config.gp_length_scale),
        "gp_optimize_length_scale": bool(config.gp_optimize_length_scale),
        "n_measurements_used": int(len(grid.measurements)),
        "n_discretization_points": int(np.count_nonzero(grid.free_mask)),
        "estimation_runtime_sec": float(estimation_runtime_sec),
    }
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    if verbose:
        sample_label = sample_size if sample_size is not None else "all"
        print(f"Saved GP estimate for {sample_csv.name} ({sample_label} samples) to {estimate_path}")

    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="run_gp_wind",
        description=(
            "Run the GP wind estimator for one sample CSV or all sample_points*.csv files "
            "of a scenario."
        ),
    )
    parser.add_argument(
        "samples",
        type=str,
        nargs="+",
        help="Path(s) to sample CSV files, scenario directories, or scenario.yaml files.",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for raw_input in args.samples:
        input_path = Path(raw_input).resolve(strict=True)
        config = ScenarioConfig.load(infer_scenario_path(input_path))
        sample_files = select_sample_files(input_path, config)

        sample_sizes = config.wind_sample_sizes or (None,)
        for sample_size in sample_sizes:
            for sample_csv in sample_files:
                if args.verbose:
                    label = f" with first {sample_size} samples" if sample_size is not None else ""
                    print(f"Running GP wind for {sample_csv.name}{label}")
                run_case(config, sample_csv, sample_size, args.verbose)


if __name__ == "__main__":
    main()
