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
import yaml
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
        self.get_logger().info("Waiting for services...")
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


def load_free_space_mask(map_yaml_file: Path) -> tuple[np.ndarray, float, float, float]:
    with Path(map_yaml_file).open("r") as f:
        data = yaml.safe_load(f)

    image_name = data.get("image")
    origin = data.get("origin")
    resolution = data.get("resolution")
    free_thresh = float(data.get("free_thresh", 0.1))
    negate = int(data.get("negate", 0))
    if image_name is None or not isinstance(origin, list) or len(origin) < 2 or resolution is None:
        raise ValueError(f"Invalid map yaml: {map_yaml_file}")

    image_path = Path(map_yaml_file).parent / image_name
    with image_path.open("r") as f:
        tokens = [token for line in f for token in line.split() if not line.startswith("#")]
    if len(tokens) < 4 or tokens[0] != "P2":
        raise ValueError(f"Unsupported occupancy image format: {image_path}")
    width = int(tokens[1])
    height = int(tokens[2])
    max_value = float(tokens[3])
    image = np.asarray(tokens[4:], dtype=float).reshape(height, width)
    if max_value > 0:
        image = image * (255.0 / max_value)

    occupancy = image / 255.0 if negate else (255.0 - image) / 255.0
    free_mask = occupancy < free_thresh
    return free_mask, float(resolution), float(origin[0]), float(origin[1])


def save_estimation_csv(
    output_path: Path,
    res: WindEstimation.Response,
    cell_size: float,
    map_yaml_file: Path | str,
) -> None:
    n = len(res.u)
    if res.map_width <= 0 or n == 0 or len(res.v) != n or n % res.map_width != 0:
        raise ValueError("Invalid map size in WindEstimation response")
    map_yaml_file = Path(map_yaml_file)
    free_mask, map_resolution, origin_x, origin_y = load_free_space_mask(map_yaml_file)
    gmrf_origin_x = cell_size * round(origin_x / cell_size)
    gmrf_origin_y = cell_size * round(origin_y / cell_size)
    height, width = free_mask.shape

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_id", "x", "y", "z", "wind_x", "wind_y", "wind_z"])
        for idx in range(n):
            grid_x = idx % res.map_width
            grid_y = idx // res.map_width
            x = gmrf_origin_x + (grid_x + 0.5) * cell_size
            y = gmrf_origin_y + (grid_y + 0.5) * cell_size
            x_idx = int((x - origin_x) / map_resolution)
            y_idx = int((y - origin_y) / map_resolution)
            if x_idx < 0 or x_idx >= width or y_idx < 0 or y_idx >= height:
                continue
            if not free_mask[height - 1 - y_idx, x_idx]:
                continue
            z = 1.0
            writer.writerow([idx, x, y, z, res.u[idx], res.v[idx], 0.0])

def save_estimation_png(
    output_path: Path,
    res: WindEstimation.Response,
    used_samples: int,
    observations: list[Observation] | None = None,
    cell_size: float | None = None,
    map_yaml_file: Path | str | None = None,
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

    if cell_size is not None and map_yaml_file is not None:
        free_mask, map_resolution, origin_x, origin_y = load_free_space_mask(Path(map_yaml_file))
        gmrf_origin_x = cell_size * round(origin_x / cell_size)
        gmrf_origin_y = cell_size * round(origin_y / cell_size)
        x_plot = gmrf_origin_x + (xx + 0.5) * cell_size
        y_plot = gmrf_origin_y + (yy + 0.5) * cell_size
    else:
        free_mask = None
        map_resolution = None
        origin_x = 0.0
        origin_y = 0.0
        x_plot = xx
        y_plot = yy

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5), dpi=160)

    if free_mask is not None and map_resolution is not None:
        occupied_image = np.zeros((*free_mask.shape, 4), dtype=float)
        occupied_image[np.flipud(~free_mask)] = (0.45, 0.45, 0.45, 0.25)
        occ_height, occ_width = free_mask.shape
        ax.imshow(
            occupied_image,
            origin="lower",
            extent=(
                origin_x,
                origin_x + occ_width * map_resolution,
                origin_y,
                origin_y + occ_height * map_resolution,
            ),
            interpolation="nearest",
            zorder=0,
        )

    if streamplot:
        stream = ax.streamplot(
            x_plot[0, :],
            y_plot[:, 0],
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
            x_plot[::plot_step, ::plot_step],
            y_plot[::plot_step, ::plot_step],
            u_arr[::plot_step, ::plot_step],
            v_arr[::plot_step, ::plot_step],
            speed[::plot_step, ::plot_step],
            cmap="coolwarm",
            angles="xy",
            scale_units="xy",
            scale=None,
            width=0.0022,
            pivot="mid",
            zorder=2,
        )

    cbar = fig.colorbar(mappable, ax=ax, pad=0.02)
    cbar.set_label("Magnitude")

    if observations:
        if cell_size is not None and map_yaml_file is not None:
            obs_x = [obs.x for obs in observations]
            obs_y = [obs.y for obs in observations]
        else:
            obs_x = [obs.x for obs in observations]
            obs_y = [obs.y for obs in observations]
        ax.scatter(
            obs_x,
            obs_y,
            s=14,
            c="black",
            alpha=0.9,
            marker="o",
            linewidths=0.3,
            edgecolors="white",
            label="Measurements",
            zorder=3,
        )
        ax.legend(loc="upper right")

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [m]" if cell_size is not None and map_yaml_file is not None else "x")
    ax.set_ylabel("y [m]" if cell_size is not None and map_yaml_file is not None else "y")
    ax.set_title(f"GMRF Wind Estimation based on {used_samples} samples")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)

    print(f"[INFO] Saved estimation plot: {output_path} ({res.map_width}x{map_height})")
