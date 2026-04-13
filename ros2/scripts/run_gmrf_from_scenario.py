from dataclasses import dataclass, field
from pathlib import Path
import argparse
import json
import subprocess

import gmrf_client
import numpy as np
import pandas as pd
import rclpy
import yaml
from scipy.spatial import cKDTree

SINGLE_LAYER_Z_SPAN = 0.1


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


@dataclass(frozen=True)
class GmrfScenarioConfig:
    name: str
    occupancy_yaml: Path
    occupancy_image: Path
    wind_csv: Path
    sample_dir: Path
    result_dir: Path
    z_height: float | None = None
    z_tol: float = 0.05
    max_xy_dist: float = 0.2
    gmrf_cell_size: float = 0.25
    variance_speed: float = 0.01
    variance_direction: float = 0.01
    batch_size: int = 50
    sample_sizes: list[int] = field(default_factory=list)


def load_scenario(scenario: Path) -> GmrfScenarioConfig:
    if scenario.is_dir():
        scenario = scenario / "scenario.yaml"
    with open(str(scenario), "r") as f:
        try:
            yml_content = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            raise RuntimeError(f"Failed to parse {scenario}") from exc

        root = scenario.parent
        geometry = yml_content.get("geometry", {})
        gt_slicing = yml_content.get("ground_truth_slicing", {})
        gmrf_params = yml_content.get("gmrf_parameters", {})
        data = yml_content.get("data", {})

        return GmrfScenarioConfig(
            name=yml_content.get("name"),
            occupancy_yaml=Path(root / geometry.get("occupancy_yaml")),
            occupancy_image=Path(root / geometry.get("occupancy_image")),
            wind_csv=Path(root / data.get("wind_csv")),
            sample_dir=Path(root / data.get("sample_dir")),
            result_dir=Path(root / data.get("result_dir")),
            z_height=gt_slicing.get("z_height", GmrfScenarioConfig.z_height),
            z_tol=gt_slicing.get("z_tol", GmrfScenarioConfig.z_tol),
            max_xy_dist=gt_slicing.get("max_xy_dist", GmrfScenarioConfig.max_xy_dist),
            gmrf_cell_size=gmrf_params.get("cell_size", GmrfScenarioConfig.gmrf_cell_size),
            variance_speed=gmrf_params.get("var_speed", GmrfScenarioConfig.variance_speed),
            variance_direction=gmrf_params.get("var_direction", GmrfScenarioConfig.variance_direction),
            batch_size=gmrf_params.get("batch_size", GmrfScenarioConfig.batch_size),
            sample_sizes=yml_content.get("wind_sample_sizes", []),
        )


def infer_z_height(wind_csv: Path, z_height: float | None) -> float:
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


def load_ground_truth_layer(config: GmrfScenarioConfig) -> tuple[pd.DataFrame, float]:
    z_height = infer_z_height(config.wind_csv, config.z_height)
    gt = pd.read_csv(
        config.wind_csv,
        usecols=["Points:0", "Points:1", "Points:2", "U:0", "U:1"],
    ).rename(
        columns={
            "Points:0": "x",
            "Points:1": "y",
            "Points:2": "z",
            "U:0": "wind_x",
            "U:1": "wind_y",
        }
    )
    layer = gt.loc[np.abs(gt["z"].to_numpy(dtype=float) - z_height) <= config.z_tol]
    if layer.empty:
        raise ValueError(f"No ground-truth rows found for z={z_height}, z_tol={config.z_tol}")
    return layer.reset_index(drop=True), z_height


def vector_rmse(true_uv: np.ndarray, est_uv: np.ndarray) -> float:
    diff = est_uv - true_uv
    return float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))


def directional_rmse(true_uv: np.ndarray, est_uv: np.ndarray, eps: float = 1e-12) -> float:
    true_norm = np.linalg.norm(true_uv, axis=1)
    est_norm = np.linalg.norm(est_uv, axis=1)
    mask = (true_norm > eps) & (est_norm > eps)
    if not np.any(mask):
        return 0.0
    true_dir = true_uv[mask] / true_norm[mask, None]
    est_dir = est_uv[mask] / est_norm[mask, None]
    diff = est_dir - true_dir
    return float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))


def angular_rmse_deg(true_uv: np.ndarray, est_uv: np.ndarray, eps: float = 1e-12) -> float:
    true_norm = np.linalg.norm(true_uv, axis=1)
    est_norm = np.linalg.norm(est_uv, axis=1)
    mask = (true_norm > eps) & (est_norm > eps)
    if not np.any(mask):
        return 0.0
    dots = np.sum(true_uv[mask] * est_uv[mask], axis=1) / (true_norm[mask] * est_norm[mask])
    angles_deg = np.degrees(np.arccos(np.clip(dots, -1.0, 1.0)))
    return float(np.sqrt(np.mean(angles_deg**2)))


def compute_metrics(
    config: GmrfScenarioConfig,
    estimate_csv: Path,
    sample_csv: Path,
    sample_size: int,
) -> dict:
    result = pd.read_csv(estimate_csv, usecols=["x", "y", "wind_x", "wind_y"])
    if result.empty:
        raise ValueError(f"No estimated wind rows found in {estimate_csv}")

    ground_truth, z_height = load_ground_truth_layer(config)
    result_xy = result[["x", "y"]].to_numpy(dtype=float)
    result_uv = result[["wind_x", "wind_y"]].to_numpy(dtype=float)
    gt_xy = ground_truth[["x", "y"]].to_numpy(dtype=float)
    gt_uv_all = ground_truth[["wind_x", "wind_y"]].to_numpy(dtype=float)

    dist, idx = cKDTree(gt_xy).query(result_xy, workers=-1)
    keep = dist <= config.max_xy_dist
    if not np.any(keep):
        raise ValueError(
            f"No GMRF result points matched ground truth within max_xy_dist={config.max_xy_dist}"
        )

    true_uv = gt_uv_all[idx[keep]]
    est_uv = result_uv[keep]

    return {
        "scenario": config.name,
        "estimator": "gmrf",
        "sample_name": sample_csv.stem,
        "sample_size": sample_size,
        "samples_csv": str(sample_csv),
        "wind_csv": str(config.wind_csv),
        "occupancy_yaml": str(config.occupancy_yaml),
        "gmrf_cell_size": float(config.gmrf_cell_size),
        "variance_speed": float(config.variance_speed),
        "variance_direction": float(config.variance_direction),
        "batch_size": int(config.batch_size),
        "z_height": float(z_height),
        "z_tol": float(config.z_tol),
        "n_discretization_points": int(len(result_xy)),
        "rmse": vector_rmse(true_uv, est_uv),
        "directional_rmse": directional_rmse(true_uv, est_uv),
        "angular_rmse_deg": angular_rmse_deg(true_uv, est_uv),
        "n_result_points": int(len(result_xy)),
        "n_evaluated_points": int(np.count_nonzero(keep)),
        "n_dropped_unmatched_gt": int(np.count_nonzero(~keep)),
        "max_xy_dist": float(np.max(dist[keep])),
        "mean_xy_dist": float(np.mean(dist[keep])),
    }


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


def run_case(config: GmrfScenarioConfig, sample_file_csv: Path, sample_size: int,
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

    res = node.query_estimation()
    if res is None:
        node.get_logger().error("WindEstimation query failed.")
        return 1

    out_csv = result_dir / "wind_estimate_gmrf.csv"
    out_png = result_dir / "wind_estimate_gmrf.png"
    metrics_path = result_dir / "metrics.json"
    gmrf_client.save_estimation_csv(out_csv, res, config.gmrf_cell_size, config.occupancy_yaml)
    gmrf_client.save_estimation_png(out_png, res, sample_size, obs_to_use, config.gmrf_cell_size, config.occupancy_yaml)

    metrics = compute_metrics(config, out_csv, sample_file_csv, sample_size)
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    node.clear_observations()
    print(f"Saved result to {out_csv}")
    print(f"Saved metrics to {metrics_path}")
    return 0


def main() -> int:
    args = parse()
    scenario_cfg = load_scenario(Path(args.scenario).resolve())

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

        for sample_size in scenario_cfg.sample_sizes:
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
