#!/usr/bin/env python3
"""
Run GMRF estimation for all sample CSV files in one folder and save one result CSV per input file.

This script is the main workflow runner. It reads every CSV file from `CSV_DIR`, clears old
observations in the GMRF node, sends the new observations in small batches, requests the
estimated wind field, and writes the result to `OUTPUT_DIR`. Each input CSV is processed
independently, so results from earlier files are not kept for later files.
"""
import sys
from pathlib import Path

import rclpy
import yaml

import gmrf_client

CSV_DIR = Path("/app/csv_wind_data/10x6_central_obstacle/csv_wind_sample_sets")
OUTPUT_DIR = Path("/app/ros2/results")
PARAMS_FILE = Path("/app/ros2/src/GMRF-wind/gmrf_wind_mapping/launch/gmrf_comparison.params.yaml")
BATCH_SIZE = 25
VAR_SPEED = 1e-2
VAR_DIRECTION = 1e-2


def load_gmrf_params(params_file: Path) -> tuple[Path, float]:
    with params_file.open("r") as f:
        data = yaml.safe_load(f)

    for node_config in data.values():
        ros_params = node_config.get("ros__parameters", {})
        map_yaml_file = ros_params.get("map_yaml_file")
        cell_size = ros_params.get("cell_size")
        if map_yaml_file is not None and cell_size is not None:
            return Path(map_yaml_file), float(cell_size)

    raise KeyError(f"map_yaml_file/cell_size not found in {params_file}")


def main() -> int:
    if BATCH_SIZE <= 0:
        raise ValueError("[ERROR] BATCH_SIZE must be > 0")
    try:
        csv_files = sorted(p for p in CSV_DIR.glob("*.csv") if p.is_file())
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {CSV_DIR}")
        map_yaml_file, cell_size = load_gmrf_params(PARAMS_FILE)
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2

    rclpy.init()
    node = gmrf_client.GmrfClient()
    try:
        if not node.wait_for_services(gmrf_client.WAIT_SERVICE_SEC):
            node.get_logger().error("Required GMRF services not available")
            return 1

        total_files = len(csv_files)
        for idx, csv_path in enumerate(csv_files, start=1):
            node.get_logger().info(f"[{idx}/{total_files}] Processing {csv_path}")

            if not node.clear_observations(timeout_sec=gmrf_client.CALL_TIMEOUT_SEC):
                node.get_logger().error("ClearObservations failed")
                return 1

            try:
                observations = gmrf_client.load_observations(csv_path, VAR_SPEED, VAR_DIRECTION)
            except Exception as exc:
                node.get_logger().error(f"Failed to load CSV {csv_path}: {exc}")
                return 1

            if not observations:
                node.get_logger().error(f"No observations in {csv_path}")
                return 1

            sent = 0
            for batch_idx, chunk in enumerate(gmrf_client.batched(observations, BATCH_SIZE), 1):
                if not node.send_batch(chunk, timeout_sec=gmrf_client.CALL_TIMEOUT_SEC):
                    node.get_logger().error(
                        f"AddWindObservation failed at file {csv_path}, batch {batch_idx}"
                    )
                    return 1
                sent += len(chunk)
                node.get_logger().info(f"  batch {batch_idx} sent ({sent}/{len(observations)})")

            res = node.query_estimation(timeout_sec=gmrf_client.CALL_TIMEOUT_SEC)
            if res is None:
                node.get_logger().error(f"WindEstimation query failed for {csv_path}")
                return 1
            
            out_csv = OUTPUT_DIR / f"cell{cell_size}" / f"{csv_path.stem}_gmrf_{len(res.u)}nodes.csv"
            out_png = out_csv.with_suffix(".png")

            gmrf_client.save_estimation_csv(out_csv, res, cell_size, map_yaml_file)
            gmrf_client.save_estimation_png(out_png, res, sent, streamplot=False)

        node.get_logger().info(f"Done. Processed {total_files} CSV files.")
        return 0
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
