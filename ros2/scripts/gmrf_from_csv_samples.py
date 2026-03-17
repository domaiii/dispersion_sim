#!/usr/bin/env python3
"""
Run GMRF estimation for prefix subsets of the stored sample CSV files.

Each sample CSV file is treated as one canonical random ordering. Smaller sample sizes are
created in-memory by taking the first N observations from that file, so all estimators can
share identical nested subsets.
"""
import sys
import re
from pathlib import Path

import rclpy
import yaml

import gmrf_client

CSV_DIR = Path("/app/csv_wind_data/10x6_central_obstacle/csv_wind_sample_sets")
OUTPUT_DIR = Path("/app/ros2/results")
LAUNCH_DIR = Path("/app/ros2/src/GMRF-wind/gmrf_wind_mapping/launch")
LAUNCH_MODULE = "gmrf_comparison_launch"
BATCH_SIZE = 25
VAR_SPEED = 1e-2
VAR_DIRECTION = 1e-2
SUBSET_SIZES = [25, 50, 100, 200, 400]


sys.path.insert(0, str(LAUNCH_DIR))
launch_config = __import__(LAUNCH_MODULE)


def load_gmrf_params() -> tuple[Path, float]:
    params = launch_config.GMRF_PARAMS
    return Path(params["map_yaml_file"]), float(params["cell_size"])


def discover_base_csv_files() -> list[Path]:
    base_files = sorted(CSV_DIR.glob("sample_points*.csv"))
    if not base_files:
        raise FileNotFoundError(f"No base sample files matching sample_points*.csv in {CSV_DIR}")
    return base_files


def extract_seed(csv_path: Path) -> int:
    match = re.search(r"_seed(\d+)", csv_path.stem)
    if match is None:
        raise ValueError(f"Could not extract seed from CSV filename: {csv_path.name}")
    return int(match.group(1))


def main() -> int:
    base_csv_files = discover_base_csv_files()
    map_yaml_file, cell_size = load_gmrf_params()

    rclpy.init()
    node = gmrf_client.GmrfClient()
    try:
        if not node.wait_for_services(gmrf_client.WAIT_SERVICE_SEC):
            node.get_logger().error("Required GMRF services not available")
            return 1

        total_runs = len(base_csv_files) * len(SUBSET_SIZES)
        run_idx = 0
        for csv_path in base_csv_files:
            try:
                all_observations = gmrf_client.load_observations(csv_path, VAR_SPEED, VAR_DIRECTION)
                seed = extract_seed(csv_path)
            except Exception as exc:
                node.get_logger().error(f"Failed to load base CSV {csv_path}: {exc}")
                return 1

            required_samples = max(SUBSET_SIZES)
            if len(all_observations) < required_samples:
                node.get_logger().error(
                    f"Base CSV {csv_path} has only {len(all_observations)} rows, expected at least {required_samples}"
                )
                return 1

            for subset_size in SUBSET_SIZES:
                run_idx += 1
                observations = all_observations[:subset_size]
                run_name = f"gmrf_result_n{subset_size}_seed{seed}"
                node.get_logger().info(f"[{run_idx}/{total_runs}] Processing {run_name} from {csv_path.name}")

                if not node.clear_observations(timeout_sec=gmrf_client.CALL_TIMEOUT_SEC):
                    node.get_logger().error("ClearObservations failed")
                    return 1

                sent = 0
                for batch_idx, chunk in enumerate(gmrf_client.batched(observations, BATCH_SIZE), 1):
                    if not node.send_batch(chunk, timeout_sec=gmrf_client.CALL_TIMEOUT_SEC):
                        node.get_logger().error(
                            f"AddWindObservation failed at {run_name}, batch {batch_idx}"
                        )
                        return 1
                    sent += len(chunk)
                    node.get_logger().info(f"  batch {batch_idx} sent ({sent}/{len(observations)})")

                res = node.query_estimation(timeout_sec=gmrf_client.CALL_TIMEOUT_SEC)
                if res is None:
                    node.get_logger().error(f"WindEstimation query failed for {run_name}")
                    return 1

                out_csv = OUTPUT_DIR / f"cell{cell_size}" / f"{run_name}_{len(res.u)}nodes.csv"
                out_png = out_csv.with_suffix(".png")

                gmrf_client.save_estimation_csv(out_csv, res, cell_size, map_yaml_file)
                gmrf_client.save_estimation_png(
                    out_png,
                    res,
                    sent,
                    observations=observations,
                    cell_size=cell_size,
                    map_yaml_file=map_yaml_file,
                    streamplot=False,
                )

        node.get_logger().info(f"Done. Processed {total_runs} GMRF runs from {len(base_csv_files)} base sample files.")
        return 0
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
