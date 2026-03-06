#!/usr/bin/env python3
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import rclpy
from rclpy.node import Node

from gmrf_msgs.srv import WindEstimation


@dataclass(frozen=True)
class ExportConfig:
    estimation_service: str = "/WindEstimation"
    wait_service_sec: float = 30.0
    query_timeout_sec: float = 20.0
    output_png: Path | None = Path("/app/ros2/results/gmrf_estimation.png")
    plot_step: int = 2

CONFIG = ExportConfig()

class WindEstimationClient(Node):
    def __init__(self, service_name: str) -> None:
        super().__init__("wind_estimation_export_client")
        self._client = self.create_client(WindEstimation, service_name)

    def wait_for_service(self, timeout_sec: float) -> bool:
        return self._client.wait_for_service(timeout_sec=timeout_sec)

    def query_full_estimation(self, timeout_sec: float) -> WindEstimation.Response | None:
        req = WindEstimation.Request()
        req.x = []
        req.y = []

        future = self._client.call_async(req)
        end_t = time.time() + timeout_sec
        while rclpy.ok() and not future.done():
            if time.time() > end_t:
                return None
            rclpy.spin_once(self, timeout_sec=0.1)

        if future.cancelled() or future.exception() is not None:
            return None
        return future.result()

def save_estimation_png(
    output_path: Path,
    map_width: int,
    u: list[float],
    v: list[float],
    plot_step: int,
) -> None:
    if plot_step <= 0:
        raise ValueError("CONFIG.plot_step must be > 0")

    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as exc:
        raise RuntimeError(f"matplotlib/numpy not available: {exc}") from exc

    n = len(u)
    if map_width <= 0 or n == 0 or n % map_width != 0:
        raise ValueError("Invalid map size in WindEstimation response")
    map_height = n // map_width

    u_arr = np.asarray(u, dtype=float).reshape(map_height, map_width)
    v_arr = np.asarray(v, dtype=float).reshape(map_height, map_width)
    speed = np.hypot(u_arr, v_arr)
    yy, xx = np.mgrid[0:map_height, 0:map_width]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5), dpi=160)

    quiv = ax.quiver(
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

    cbar = fig.colorbar(quiv, ax=ax, pad=0.02)
    cbar.set_label("Magnitude")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("GMRF Wind Estimation")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)

    print(f"[INFO] Saved estimation plot: {output_path} ({map_width}x{map_height})")


def export_estimation(config: ExportConfig = CONFIG) -> int:
    if config.output_png is None:
        print("[ERROR] config.output_png is None", file=sys.stderr)
        return 2

    started_here = not rclpy.ok()
    if started_here:
        rclpy.init()

    node = WindEstimationClient(config.estimation_service)
    try:
        if not node.wait_for_service(config.wait_service_sec):
            node.get_logger().error(
                f"Service '{config.estimation_service}' not available "
                f"after {config.wait_service_sec:.1f}s"
            )
            return 1

        res = node.query_full_estimation(config.query_timeout_sec)
        if res is None:
            node.get_logger().error(
                f"WindEstimation query failed or timed out ({config.query_timeout_sec:.1f}s)"
            )
            return 1

        if config.output_png is not None:
            save_estimation_png(
                config.output_png,
                int(res.map_width),
                list(res.u),
                list(res.v),
                config.plot_step,
            )

        return 0
    finally:
        node.destroy_node()
        if started_here:
            rclpy.shutdown()


def main() -> int:
    return export_estimation(CONFIG)


if __name__ == "__main__":
    raise SystemExit(main())
