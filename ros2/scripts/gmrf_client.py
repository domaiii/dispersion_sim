#!/usr/bin/env python3
"""
Shared GMRF helper code for talking to the ROS2 services and reading or writing CSV data.

This module defines the observation data model, wraps the three GMRF ROS2 services in a small
client class, converts CSV rows into wind observations, splits observations into batches, and
saves estimation results back to CSV. It is imported by the executable scripts so the actual
workflow code can stay short.
"""
import csv
import math
import time
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import rclpy
from rclpy.node import Node

from gmrf_msgs.srv import AddWindObservation, ClearObservations, WindEstimation

ADD_OBSERVATION_SERVICE = "/AddWindObservation"
CLEAR_OBSERVATION_SERVICE = "/ClearObservations"
ESTIMATION_SERVICE = "/WindEstimation"
WAIT_SERVICE_SEC = 30.0
CALL_TIMEOUT_SEC = 5.0


@dataclass(frozen=True)
class Observation:
    x: float
    y: float
    speed: float
    direction: float
    var_speed: float
    var_direction: float


class GmrfClient(Node):
    def __init__(
        self,
        add_service: str = ADD_OBSERVATION_SERVICE,
        clear_service: str = CLEAR_OBSERVATION_SERVICE,
        estimation_service: str = ESTIMATION_SERVICE,
    ) -> None:
        super().__init__("gmrf_script_client")
        self._add_client = self.create_client(AddWindObservation, add_service)
        self._clear_client = self.create_client(ClearObservations, clear_service)
        self._est_client = self.create_client(WindEstimation, estimation_service)

    def wait_for_services(self, timeout_sec: float = WAIT_SERVICE_SEC) -> bool:
        ok_add = self._add_client.wait_for_service(timeout_sec=timeout_sec)
        ok_clear = self._clear_client.wait_for_service(timeout_sec=timeout_sec)
        ok_est = self._est_client.wait_for_service(timeout_sec=timeout_sec)
        return ok_add and ok_clear and ok_est

    def _wait_future(self, future, timeout_sec: float):
        end_t = time.time() + timeout_sec
        while rclpy.ok() and not future.done():
            if time.time() > end_t:
                return None
            rclpy.spin_once(self, timeout_sec=0.1)
        if future.cancelled() or future.exception() is not None:
            return None
        return future.result()

    def clear_observations(self, timeout_sec: float = CALL_TIMEOUT_SEC) -> bool:
        future = self._clear_client.call_async(ClearObservations.Request())
        result = self._wait_future(future, timeout_sec)
        return bool(result and result.success)

    def send_batch(self, observations: list[Observation], timeout_sec: float = CALL_TIMEOUT_SEC) -> bool:
        req = AddWindObservation.Request()
        req.wind_speed = [obs.speed for obs in observations]
        req.wind_direction = [obs.direction for obs in observations]
        req.var_speed = [obs.var_speed for obs in observations]
        req.var_direction = [obs.var_direction for obs in observations]
        req.x_pos = [obs.x for obs in observations]
        req.y_pos = [obs.y for obs in observations]

        future = self._add_client.call_async(req)
        return self._wait_future(future, timeout_sec) is not None

    def query_estimation(self, timeout_sec: float = CALL_TIMEOUT_SEC) -> WindEstimation.Response | None:
        req = WindEstimation.Request()
        req.x = []
        req.y = []
        future = self._est_client.call_async(req)
        return self._wait_future(future, timeout_sec)


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


def save_estimation_csv(
    output_path: Path,
    res: WindEstimation.Response
) -> None:
    n = len(res.u)
    if res.map_width <= 0 or n == 0 or len(res.v) != n or n % res.map_width != 0:
        raise ValueError("Invalid map size in WindEstimation response")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_id", "x", "y", "z", "wind_x", "wind_y", "wind_z"])
        for idx in range(n):
            x = idx % res.map_width
            y = idx // res.map_width
            writer.writerow([idx, x, y, 0.0, res.u[idx], res.v[idx], 0.0])

def save_estimation_png(
    output_path: Path,
    res: WindEstimation.Response,
    used_samples: int,
    streamplot: bool = False,
    plot_step: int = 1
) -> None:
    n = len(res.u)
    if res.map_width <= 0 or n == 0 or n % res.map_width != 0:
        raise ValueError("Invalid map size in WindEstimation response")
    map_height = n // res.map_width

    u_arr = np.asarray(res.u, dtype=float).reshape(map_height, res.map_width)
    v_arr = np.asarray(res.v, dtype=float).reshape(map_height, res.map_width)
    speed = np.hypot(u_arr, v_arr)
    yy, xx = np.mgrid[0:map_height, 0:res.map_width]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5), dpi=160)

    if streamplot:
        stream = ax.streamplot(
            np.arange(res.map_width),
            np.arange(map_height),
            u_arr,
            v_arr,
            color=speed,
            cmap="coolwarm",
            density=1.1,
            linewidth=1.0,
            arrowsize=0.9,
        )
        mappable = stream.lines
    else:
        mappable = ax.quiver(
            xx[::plot_step, ::plot_step],
            yy[::plot_step, ::plot_step],
            u_arr[::plot_step, ::plot_step],
            v_arr[::plot_step, ::plot_step],
            speed[::plot_step, ::plot_step],
            cmap="coolwarm",
            angles="xy",
            scale_units="xy",
            scale=None,
            width=0.0022,
            pivot="mid",
        )

    cbar = fig.colorbar(mappable, ax=ax, pad=0.02)
    cbar.set_label("Magnitude")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"GMRF Wind Estimation based on {used_samples} samples")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)

    print(f"[INFO] Saved estimation plot: {output_path} ({res.map_width}x{map_height})")
