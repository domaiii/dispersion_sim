#!/usr/bin/env python3
import csv
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import rclpy
from rclpy.node import Node

from gmrf_msgs.srv import AddWindObservation
from export_gmrf_estimation import export_estimation


DEFAULT_SAMPLE_CSV = Path(
    "/app/csv_wind_data/10x6_central_obstacle/csv_wind_sample_sets/sample_points_n100_seed42.csv"
)


@dataclass(frozen=True)
class FeedConfig:
    csv_path: Path = DEFAULT_SAMPLE_CSV
    add_observation_service: str = "/AddWindObservation"
    batch_size: int = 25
    var_speed: float = 0.001
    var_direction: float = 0.0001
    wait_service_sec: float = 30.0
    call_timeout_sec: float = 10.0
    sleep_between_batches_sec: float = 0.0
    run_export_after_feed: bool = True


@dataclass(frozen=True)
class Observation:
    x: float
    y: float
    speed: float
    direction: float
    var_speed: float
    var_direction: float


CONFIG = FeedConfig()


class CsvObservationClient(Node):
    def __init__(self, service_name: str) -> None:
        super().__init__("csv_observation_client")
        self._client = self.create_client(AddWindObservation, service_name)

    def wait_for_service(self, timeout_sec: float) -> bool:
        return self._client.wait_for_service(timeout_sec=timeout_sec)

    def send_batch(self, observations: list[Observation], timeout_sec: float) -> bool:
        req = AddWindObservation.Request()
        req.wind_speed = [obs.speed for obs in observations]
        req.wind_direction = [obs.direction for obs in observations]
        req.var_speed = [obs.var_speed for obs in observations]
        req.var_direction = [obs.var_direction for obs in observations]
        req.x_pos = [obs.x for obs in observations]
        req.y_pos = [obs.y for obs in observations]

        future = self._client.call_async(req)
        end_t = time.time() + timeout_sec
        while rclpy.ok() and not future.done():
            if time.time() > end_t:
                return False
            rclpy.spin_once(self, timeout_sec=0.1)

        if future.cancelled():
            return False
        if future.exception() is not None:
            self.get_logger().error(f"Service call failed: {future.exception()}")
            return False
        return True


def load_observations(csv_path: Path, var_speed: float, var_direction: float) -> list[Observation]:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    observations: list[Observation] = []
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        required = {"x", "y", "wind_x", "wind_y"}
        missing = required.difference(set(reader.fieldnames or []))
        if missing:
            raise ValueError(f"Missing required CSV columns: {sorted(missing)}")

        for row in reader:
            wx = float(row["wind_x"])
            wy = float(row["wind_y"])
            observations.append(
                Observation(
                    x=float(row["x"]),
                    y=float(row["y"]),
                    speed=math.hypot(wx, wy),
                    direction=math.atan2(wy, wx),
                    var_speed=var_speed,
                    var_direction=var_direction,
                )
            )
    return observations


def batched(data: list[Observation], size: int):
    for i in range(0, len(data), size):
        yield data[i : i + size]


def main() -> int:
    try:
        observations = load_observations(CONFIG.csv_path, CONFIG.var_speed, CONFIG.var_direction)
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2

    if not observations:
        print("[ERROR] CSV contains no observations", file=sys.stderr)
        return 2
    if CONFIG.batch_size <= 0:
        print("[ERROR] CONFIG.batch_size must be > 0", file=sys.stderr)
        return 2

    rclpy.init()
    node = CsvObservationClient(CONFIG.add_observation_service)
    try:
        if not node.wait_for_service(CONFIG.wait_service_sec):
            node.get_logger().error(
                f"Service '{CONFIG.add_observation_service}' not available "
                f"after {CONFIG.wait_service_sec:.1f}s"
            )
            return 1

        total = len(observations)
        sent = 0
        for idx, chunk in enumerate(batched(observations, CONFIG.batch_size), start=1):
            ok = node.send_batch(chunk, timeout_sec=CONFIG.call_timeout_sec)
            if not ok:
                node.get_logger().error(f"Batch {idx} failed")
                return 1

            sent += len(chunk)
            node.get_logger().info(f"Batch {idx} sent ({sent}/{total})")
            if CONFIG.sleep_between_batches_sec > 0.0:
                time.sleep(CONFIG.sleep_between_batches_sec)

        node.get_logger().info(f"Done. Sent {sent} observations from {CONFIG.csv_path}")
        if CONFIG.run_export_after_feed:
            node.get_logger().info("Running export step")
            return export_estimation()
        return 0
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
