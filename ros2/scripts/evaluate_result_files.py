import re
from os import listdir
from pathlib import Path
from scipy.spatial import cKDTree

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS_DIR = Path("/app/ros2/results/cell0.1")
GROUND_TRUTH_CSV = Path("/app/csv_wind_data/10x6_central_obstacle/wind_solution.csv")
GROUND_TRUTH_Z = 1.05
GROUND_TRUTH_Z_TOL = 0.02
COORD_COLUMNS = ["x", "y"]
WIND_COLUMNS = ["wind_x", "wind_y"]
MATCH_DISTANCE_THRESHOLD = 0.03


def load_ground_truth_layer() -> pd.DataFrame:
    gt = pd.read_csv(
        GROUND_TRUTH_CSV,
        usecols=["Points:0", "Points:1", "Points:2", "U:0", "U:1", "U:2"],
    ).rename(
        columns={
            "Points:0": "x",
            "Points:1": "y",
            "Points:2": "z",
            "U:0": "wind_x",
            "U:1": "wind_y",
            "U:2": "wind_z",
        }
    )

    z = gt["z"].to_numpy()
    mask = np.abs(z - GROUND_TRUTH_Z) <= GROUND_TRUTH_Z_TOL
    filtered = gt.loc[mask, COORD_COLUMNS + WIND_COLUMNS].reset_index(drop=True)
    if filtered.empty:
        raise ValueError("No ground-truth rows found for the configured z slice.")
    return filtered


def build_ground_truth_lookup(
    result_path: Path,
    ground_truth: pd.DataFrame,
    match_distance_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    result_xy = pd.read_csv(result_path, usecols=COORD_COLUMNS).to_numpy(dtype=np.float64)
    gt_xy = ground_truth[COORD_COLUMNS].to_numpy(dtype=np.float64)

    tree = cKDTree(gt_xy)
    min_distances, nearest_idx = tree.query(result_xy, workers=-1)

    keep_mask = min_distances < match_distance_threshold
    if not np.any(keep_mask):
        raise ValueError("No result points could be matched to the ground-truth layer.")

    gt_wind = ground_truth[WIND_COLUMNS].to_numpy(dtype=np.float64)[nearest_idx]
    return gt_wind, min_distances


def evaluate_file(result_path: Path, gt_wind_by_result_idx: np.ndarray, keep_mask: np.ndarray) -> float:
    result_wind = pd.read_csv(result_path, usecols=WIND_COLUMNS).to_numpy(dtype=np.float64)
    diff = result_wind[keep_mask] - gt_wind_by_result_idx[keep_mask]
    return float(np.sqrt(np.mean(np.einsum("ij,ij->i", diff, diff))))

filenames = listdir(RESULTS_DIR)
csv_files = [filename for filename in filenames if filename.endswith(".csv")]
sample_sizes = sorted(
    {
        int(match.group(1))
        for filename in csv_files
        if (match := re.search(r"_n(\d+)_", filename))
    }
)

ground_truth = load_ground_truth_layer()
first_result_path = RESULTS_DIR / csv_files[0]
ground_truth_wind, min_distances = build_ground_truth_lookup(
    first_result_path, ground_truth, MATCH_DISTANCE_THRESHOLD
)
keep_mask = min_distances < MATCH_DISTANCE_THRESHOLD

print(
    "Lookup diagnostics: "
    f"max_match_distance={float(np.max(min_distances[keep_mask])):.12f}, "
    f"mean_match_distance={float(np.mean(min_distances[keep_mask])):.12f}, "
    f"kept_cells={int(np.count_nonzero(keep_mask))}, "
    f"dropped_cells={int(np.count_nonzero(~keep_mask))}"
)

rows = []
for sample_size in sample_sizes:
    sample_files = [
        RESULTS_DIR / filename
        for filename in csv_files
        if f"_n{sample_size}_" in filename
    ]
    errors = [evaluate_file(path, ground_truth_wind, keep_mask) for path in sample_files]
    rows.append(
        {
            "sample_size": sample_size,
            "mean_rmse": float(np.mean(errors)),
            "std_rmse": float(np.std(errors)),
            "num_runs": len(errors),
        }
    )

summary = pd.DataFrame(rows).sort_values("sample_size").reset_index(drop=True)
print(summary.to_string(index=False))

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(summary["sample_size"], summary["mean_rmse"], marker="o")
ax.fill_between(
    summary["sample_size"],
    summary["mean_rmse"] - summary["std_rmse"],
    summary["mean_rmse"] + summary["std_rmse"],
    alpha=0.2,
)
ax.set_xlabel("Sample size")
ax.set_ylabel("RMSE")
ax.set_title("GMRF RMSE vs sample size")
ax.grid(True, alpha=0.3)
fig.tight_layout()
plot_path = RESULTS_DIR / "evaluation_rmse.png"
fig.savefig(plot_path, dpi=160)
print(f"\nSaved plot to {plot_path}")
