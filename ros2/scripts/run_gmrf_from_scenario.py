from pathlib import Path
import argparse
import json
import subprocess
import sys
import time

TOOLS_DIR = Path(__file__).resolve().parents[2] / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import gmrf_client
import rclpy
from scenario import ScenarioConfig

def start_gmrf_launch(config):
    return subprocess.Popen(
        [
            "ros2",
            "launch",
            "gmrf_wind_mapping",
            "gmrf_comparison_launch.py",
            f"map_yaml_file:={config.occupancy_yaml}",
            f"cell_size:={config.gmrf_cell_size}",
        ],
        text=True,
    )


def stop_gmrf_launch(process):
    if process.poll() is not None:
        return

    process.terminate()
    process.wait(timeout=30)


def count_csv_data_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8") as f:
        # Subtract the header row. Empty files should still report zero data rows.
        return max(sum(1 for _ in f) - 1, 0)


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("scenario", type=str)

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-s", "--samples", type=str, help="Path to one sample CSV.")
    group.add_argument(
        "--all-samples",
        action="store_true",
        help="Run all sample_points*.csv files in the scenario samples directory.",
    )

    return parser.parse_args()


def run_case(config: ScenarioConfig, sample_file_csv: Path, sample_size: int,
             node: gmrf_client.GmrfClient):
    result_dir = config.result_dir / "gmrf" / f"{sample_size}samples" / sample_file_csv.stem
    result_dir.mkdir(parents=True, exist_ok=True)

    observations = gmrf_client.load_observations(sample_file_csv,
                                                 config.variance_speed,
                                                 config.variance_direction)

    if len(observations) < sample_size:
        node.get_logger().error(
            f"Sample CSV file {str(sample_file_csv)} has only {len(observations)} rows, expected at least {sample_size}."
        )
        return 1

    obs_to_use = observations[:sample_size]

    print(f"Using {len(obs_to_use)}/{len(observations)} samples from {sample_file_csv}.")
    node.clear_observations()

    for batch_idx, obs_idx in enumerate(range(0, len(obs_to_use), config.batch_size)):
        batch = obs_to_use[obs_idx:obs_idx + config.batch_size]
        if not node.send_batch(batch):
            node.get_logger().error(
                f"Sending batches failed at batch {batch_idx} observations {obs_idx} - {obs_idx + config.batch_size}"
            )
            return 1

    estimation_start = time.perf_counter()
    res = node.query_estimation()
    estimation_runtime_sec = time.perf_counter() - estimation_start
    if res is None:
        node.get_logger().error("WindEstimation query failed.")
        return 1

    out_csv = result_dir / "wind_estimate_gmrf.csv"
    out_png = result_dir / "wind_estimate_gmrf.png"
    metadata_path = result_dir / "metadata_wind_est.json"
    gmrf_client.save_estimation_csv(out_csv, res, config.gmrf_cell_size, config.occupancy_yaml)
    gmrf_client.save_estimation_png(out_png, res, sample_size, obs_to_use, config.gmrf_cell_size, config.occupancy_yaml)

    metadata = {
        "scenario": config.name,
        "estimator": "gmrf",
        "sample_name": sample_file_csv.stem,
        "sample_size": sample_size,
        "samples_csv": str(sample_file_csv),
        "wind_csv": str(config.wind_csv),
        "occupancy_yaml": str(config.occupancy_yaml),
        "wind_estimate_csv": str(out_csv),
        "gmrf_cell_size": float(config.gmrf_cell_size),
        "variance_speed": float(config.variance_speed),
        "variance_direction": float(config.variance_direction),
        "batch_size": int(config.batch_size),
        "n_discretization_points": count_csv_data_rows(out_csv),
        "estimation_runtime_sec": float(estimation_runtime_sec),
    }
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    node.clear_observations()
    print(f"Saved result to {out_csv}")
    print(f"Saved metadata to {metadata_path}")
    return 0


def main() -> int:
    args = parse()
    scenario_cfg = ScenarioConfig.load(args.scenario)

    if args.samples:
        sample_files = [Path(args.samples).resolve(strict=True)]

    else: # run for all samples in scenario_name/samples
        sample_files = sorted(scenario_cfg.sample_dir.glob("sample_points*.csv"))
        if not sample_files:
            raise FileNotFoundError(f"No sample_points*.csv files found in {scenario_cfg.sample_dir}")

    launch_gmrf_core = start_gmrf_launch(scenario_cfg)
    rclpy.init()
    node = gmrf_client.GmrfClient()

    try:
        if not node.wait_for_services(gmrf_client.WAIT_SERVICE_SEC):
            node.get_logger().error("Required GMRF services not available")
            return 1

        for sample_size in scenario_cfg.wind_sample_sizes:
            for sample_file in sample_files:
                result = run_case(scenario_cfg, sample_file, sample_size, node)
                if result != 0:
                    return result
        return 0

    finally:
        node.destroy_node()
        rclpy.shutdown()
        stop_gmrf_launch(launch_gmrf_core)


if __name__ == "__main__":
    raise SystemExit(main())
