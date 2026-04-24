#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scenario import ScenarioConfig, infer_z_height


EVALUATION_METRICS = [
    "vector_rmse_m_per_s",
    "relative_l2_error",
    "magnitude_rmse_m_per_s",
    "angular_error_mean_deg",
    "angular_error_median_deg",
    "angular_error_rmse_deg",
    "estimation_runtime_sec",
]


def normalize_method_filters(methods: list[str] | None) -> set[str] | None:
    if not methods:
        return None
    return {method.strip() for method in methods if method.strip()}


def method_is_selected(
    method: str,
    include_methods: set[str] | None,
    exclude_methods: set[str] | None,
) -> bool:
    if include_methods is not None and method not in include_methods:
        return False
    if exclude_methods is not None and method in exclude_methods:
        return False
    return True


def load_ground_truth_layer(config: ScenarioConfig) -> tuple[np.ndarray, np.ndarray, float]:
    z_height = infer_z_height(config.wind_csv, config.z_height)
    gt = pd.read_csv(
        config.wind_csv,
        usecols=["Points:0", "Points:1", "Points:2", "U:0", "U:1"],
    )
    z = gt["Points:2"].to_numpy(dtype=float)
    layer = gt.loc[np.abs(z - z_height) <= config.z_tol]
    if layer.empty:
        raise ValueError(f"No ground-truth rows found for z={z_height}, z_tol={config.z_tol}")

    gt_xy = layer[["Points:0", "Points:1"]].to_numpy(dtype=float)
    gt_uv = layer[["U:0", "U:1"]].to_numpy(dtype=float)
    return gt_xy, gt_uv, z_height


def infer_method(path: Path, scenario_root: Path, payload: dict) -> str | None:
    try:
        parts = path.relative_to(scenario_root).parts
    except ValueError:
        parts = path.parts

    if "results" in parts:
        idx = parts.index("results")
        if idx + 1 < len(parts):
            return parts[idx + 1]

    if "estimator" in payload:
        return str(payload["estimator"])
    return None


def infer_sample_size(path: Path, payload: dict) -> int | None:
    if payload.get("sample_size") is not None:
        return int(payload["sample_size"])

    match = re.search(r"(\d+)samples", str(path))
    if match:
        return int(match.group(1))
    return None


def infer_estimate_csv(metrics_path: Path, method: str, payload: dict) -> Path | None:
    if payload.get("wind_estimate_csv"):
        candidate = Path(payload["wind_estimate_csv"])
        if candidate.exists():
            return candidate

    candidate = metrics_path.parent / "wind_estimate.csv"
    return candidate if candidate.exists() else None


def find_metrics_files(
    config: ScenarioConfig,
    include_methods: set[str] | None = None,
    exclude_methods: set[str] | None = None,
) -> list[Path]:
    if not config.result_dir.exists():
        return []

    files: list[Path] = []
    for method_dir in sorted(config.result_dir.iterdir()):
        if not method_dir.is_dir():
            continue
        if not method_is_selected(method_dir.name, include_methods, exclude_methods):
            continue
        files.extend(method_dir.rglob("metadata_wind_est.json"))
        files.extend(method_dir.rglob("metrics*.json"))
    return sorted(files)


def load_estimate_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path, usecols=["x", "y", "wind_x", "wind_y"])
    if df.empty:
        raise ValueError(f"No estimate rows found in {path}")
    xy = df[["x", "y"]].to_numpy(dtype=float)
    uv = df[["wind_x", "wind_y"]].to_numpy(dtype=float)
    return xy, uv


def compute_csv_error_metrics(
    estimate_csv: Path,
    gt_tree: cKDTree,
    gt_uv: np.ndarray,
    max_xy_dist: float,
    angular_speed_threshold: float,
) -> dict:
    est_xy, est_uv_all = load_estimate_csv(estimate_csv)
    dist, idx = gt_tree.query(est_xy, workers=-1)
    keep = dist <= max_xy_dist
    if not np.any(keep):
        raise ValueError(f"No estimate points in {estimate_csv} matched ground truth within {max_xy_dist}")

    est_uv = est_uv_all[keep]
    true_uv = gt_uv[idx[keep]]
    diff = est_uv - true_uv
    sq_norm = np.sum(diff * diff, axis=1)

    true_speed = np.linalg.norm(true_uv, axis=1)
    est_speed = np.linalg.norm(est_uv, axis=1)
    speed_diff = est_speed - true_speed

    denom = float(np.sqrt(np.sum(true_speed * true_speed)))
    relative_l2_error = float(np.sqrt(np.sum(sq_norm)) / denom) if denom > 1e-14 else float("nan")

    angle_mask = (true_speed > angular_speed_threshold) & (est_speed > 1e-12)
    if np.any(angle_mask):
        dots = np.sum(true_uv[angle_mask] * est_uv[angle_mask], axis=1)
        dots /= true_speed[angle_mask] * est_speed[angle_mask]
        angles_deg = np.degrees(np.arccos(np.clip(dots, -1.0, 1.0)))
        angular_error_mean_deg = float(np.mean(angles_deg))
        angular_error_median_deg = float(np.median(angles_deg))
        angular_error_rmse_deg = float(np.sqrt(np.mean(angles_deg**2)))
    else:
        angular_error_mean_deg = float("nan")
        angular_error_median_deg = float("nan")
        angular_error_rmse_deg = float("nan")

    return {
        "vector_rmse_m_per_s": float(np.sqrt(np.mean(sq_norm))),
        "relative_l2_error": relative_l2_error,
        "magnitude_rmse_m_per_s": float(np.sqrt(np.mean(speed_diff * speed_diff))),
        "angular_error_mean_deg": angular_error_mean_deg,
        "angular_error_median_deg": angular_error_median_deg,
        "angular_error_rmse_deg": angular_error_rmse_deg,
        "n_estimation_points": int(len(est_xy)),
        "n_evaluated_points": int(np.count_nonzero(keep)),
        "n_dropped_points": int(np.count_nonzero(~keep)),
        "n_angular_evaluated_points": int(np.count_nonzero(angle_mask)),
        "mean_xy_match_dist": float(np.mean(dist[keep])),
        "max_xy_match_dist": float(np.max(dist[keep])),
    }


def load_and_evaluate_runs(
    config: ScenarioConfig,
    angular_speed_threshold: float,
    include_methods: set[str] | None = None,
    exclude_methods: set[str] | None = None,
) -> pd.DataFrame:
    gt_xy, gt_uv, z_height = load_ground_truth_layer(config)
    gt_tree = cKDTree(gt_xy)

    rows = []
    skipped = []
    for metrics_path in find_metrics_files(config, include_methods, exclude_methods):
        with metrics_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        method = infer_method(metrics_path, config.root, payload)
        sample_size = infer_sample_size(metrics_path, payload)
        if method is None or sample_size is None:
            skipped.append((metrics_path, "missing method or sample_size"))
            continue
        if not method_is_selected(method, include_methods, exclude_methods):
            continue

        estimate_csv = infer_estimate_csv(metrics_path, method, payload)
        if estimate_csv is None:
            skipped.append((metrics_path, "missing estimate CSV"))
            continue

        try:
            error_metrics = compute_csv_error_metrics(
                estimate_csv,
                gt_tree,
                gt_uv,
                max_xy_dist=config.max_xy_dist,
                angular_speed_threshold=angular_speed_threshold,
            )
        except Exception as exc:
            skipped.append((metrics_path, str(exc)))
            continue

        row = dict(payload)
        row.update(error_metrics)
        row["method"] = method
        row["sample_size"] = sample_size
        row["metrics_path"] = str(metrics_path)
        row["wind_estimate_csv"] = str(estimate_csv)
        row["eval_z_height"] = float(z_height)
        row["eval_z_tol"] = float(config.z_tol)
        row["eval_max_xy_dist"] = float(config.max_xy_dist)
        row["angular_speed_threshold"] = float(angular_speed_threshold)
        rows.append(row)

    for path, reason in skipped:
        print(f"[WARN] Skipped {path}: {reason}")

    return pd.DataFrame(rows)


def aggregate_metric(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    if metric not in df.columns:
        raise ValueError(f"Metric {metric!r} not found in evaluated metrics.")

    values = df.dropna(subset=[metric]).copy()
    if values.empty:
        raise ValueError(f"Metric {metric!r} has no numeric values.")
    values[metric] = pd.to_numeric(values[metric], errors="coerce")
    values = values.dropna(subset=[metric])

    grouped = values.groupby(["method", "sample_size"], as_index=False)[metric].agg(
        mean="mean",
        std="std",
        var="var",
        count="count",
    )
    grouped["std"] = grouped["std"].fillna(0.0)
    grouped["var"] = grouped["var"].fillna(0.0)
    grouped["sem"] = grouped["std"] / np.sqrt(grouped["count"].clip(lower=1))
    grouped["metric"] = metric
    return grouped


def metric_name(metric: str) -> str:
    names = {
        "vector_rmse_m_per_s": "Vector RMSE",
        "magnitude_rmse_m_per_s": "Magnitude RMSE",
        "relative_l2_error": "Relative L2 error",
        "angular_error_mean_deg": "Mean angular error",
        "angular_error_median_deg": "Median angular error",
        "angular_error_rmse_deg": "Angular RMSE",
        "estimation_runtime_sec": "Estimation runtime",
    }
    return names.get(metric, metric.replace("_", " "))


def metric_label(metric: str, band: str) -> str:
    labels = {
        "vector_rmse_m_per_s": "Vector RMSE [m/s]",
        "magnitude_rmse_m_per_s": "Magnitude RMSE [m/s]",
        "relative_l2_error": "Relative L2 error [-]",
        "angular_error_mean_deg": "Mean angular error [deg]",
        "angular_error_median_deg": "Median angular error [deg]",
        "angular_error_rmse_deg": "Angular RMSE [deg]",
        "estimation_runtime_sec": "Estimation runtime [s]",
    }
    label = labels.get(metric, metric)
    if band != "none":
        label += f" (+/- {band})"
    return label


def plot_metric(summary: pd.DataFrame, metric: str, output_path: Path, band: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=160)

    for method, method_df in summary.groupby("method"):
        method_df = method_df.sort_values("sample_size")
        x = method_df["sample_size"].to_numpy(dtype=float)
        y = method_df["mean"].to_numpy(dtype=float)
        ax.plot(x, y, marker="o", linewidth=1.8, label=method)

        if band != "none":
            spread = method_df[band].to_numpy(dtype=float)
            ax.fill_between(x, y - spread, y + spread, alpha=0.18)

    ax.set_xlabel("Number of samples")
    ax.set_ylabel(metric_label(metric, band))
    ax.set_title(f"{metric_name(metric)} vs. number of samples")
    ax.grid(True, alpha=0.3)
    ax.legend(title="Method")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate wind scenario estimates from CSV files and aggregate metrics across random seeds."
    )
    parser.add_argument("scenario", type=str, help="Path to a scenario directory or scenario.yaml.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional output directory for evaluation CSVs and plots. Defaults to <scenario>/eval.",
    )
    parser.add_argument(
        "--band",
        choices=["std", "sem", "none"],
        default="std",
        help="Uncertainty band around the mean. Defaults to std.",
    )
    parser.add_argument(
        "--angular-speed-threshold",
        type=float,
        default=0.05,
        help="Only compute angular errors where ground-truth speed exceeds this threshold [m/s].",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=None,
        help="Only evaluate the listed method directories below results/ (for example: ns ns_variant_a).",
    )
    parser.add_argument(
        "--exclude-methods",
        nargs="+",
        default=None,
        help="Exclude the listed method directories below results/ (for example: gmrf).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = ScenarioConfig.load(args.scenario)
    include_methods = normalize_method_filters(args.methods)
    exclude_methods = normalize_method_filters(args.exclude_methods)
    output_dir = (Path(args.output_dir).expanduser().resolve() if args.output_dir else config.root / "eval")
    output_dir.mkdir(parents=True, exist_ok=True)
    df = load_and_evaluate_runs(
        config,
        angular_speed_threshold=float(args.angular_speed_threshold),
        include_methods=include_methods,
        exclude_methods=exclude_methods,
    )
    if df.empty:
        method_filter_msg = []
        if include_methods:
            method_filter_msg.append(f"including {sorted(include_methods)}")
        if exclude_methods:
            method_filter_msg.append(f"excluding {sorted(exclude_methods)}")
        method_filter_suffix = ""
        if method_filter_msg:
            method_filter_suffix = " (" + ", ".join(method_filter_msg) + ")"
        raise FileNotFoundError(
            f"No evaluable runs found below {config.result_dir}{method_filter_suffix}."
        )

    summaries = [aggregate_metric(df, metric) for metric in EVALUATION_METRICS]
    all_summary = pd.concat(summaries, ignore_index=True)

    combined_path = output_dir / "metrics_summary.csv"
    all_summary.to_csv(combined_path, index=False)
    print(f"Saved summary to: {combined_path}")

    for metric, summary in zip(EVALUATION_METRICS, summaries):
        metric_safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", metric)
        plot_path = output_dir / f"{metric_safe}_mean_by_samples.png"
        plot_metric(summary, metric, plot_path, args.band)
        print(f"Saved {metric} plot to: {plot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
