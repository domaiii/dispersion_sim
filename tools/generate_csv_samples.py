import argparse
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLS = ["Points:0", "Points:1", "Points:2", "U:0", "U:1", "U:2"]
SINGLE_LAYER_Z_SPAN = 0.1
DEFAULT_Z_TOL = 0.05


def load_wind_rows(wind_csv: str | Path) -> tuple[Path, pd.DataFrame]:
    wind_csv = Path(wind_csv).resolve()
    if not wind_csv.exists():
        raise FileNotFoundError(f"Wind CSV not found: {wind_csv}")

    df = pd.read_csv(wind_csv)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {wind_csv.name}: {missing}.")

    return wind_csv, df[REQUIRED_COLS].copy()


def select_z_slice(rows: pd.DataFrame, z_height: float | None) -> pd.DataFrame:
    z = rows["Points:2"].to_numpy(dtype=float)
    z_min = float(np.min(z))
    z_max = float(np.max(z))

    if z_height is None:
        if z_max - z_min <= SINGLE_LAYER_Z_SPAN:
            return rows
        raise ValueError(
            "Input CSV contains multiple z-levels. Pass --z-height to select a slice. "
            f"Observed z-range is [{z_min:.6g}, {z_max:.6g}]."
        )

    sliced = rows[np.abs(z - z_height) <= DEFAULT_Z_TOL]
    if len(sliced) == 0:
        raise ValueError(
            f"No rows found within +/- {DEFAULT_Z_TOL} m of z={z_height}. "
            f"Observed z-range is [{z_min:.6g}, {z_max:.6g}]."
        )
    return sliced


def sample_rows(rows: pd.DataFrame, n_points: int, seed: int) -> pd.DataFrame:
    if n_points <= 0:
        raise ValueError("n_points must be > 0.")
    if len(rows) < n_points:
        raise ValueError(
            f"Requested n_points={n_points}, but only {len(rows)} valid rows available."
        )

    rng = np.random.default_rng(seed)
    sample_idx = rng.choice(len(rows), size=n_points, replace=False)
    sampled = rows.iloc[sample_idx].reset_index(drop=True)

    return pd.DataFrame(
        {
            "sample_id": np.arange(n_points, dtype=np.int32),
            "x": sampled["Points:0"].to_numpy(),
            "y": sampled["Points:1"].to_numpy(),
            "z": sampled["Points:2"].to_numpy(),
            "wind_x": sampled["U:0"].to_numpy(),
            "wind_y": sampled["U:1"].to_numpy(),
            "wind_z": sampled["U:2"].to_numpy(),
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="generate_csv_samples",
        description="Create one or more sample CSV files from a wind ground-truth CSV.",
    )
    parser.add_argument("input_csv", type=str, help="Path to the ground-truth wind CSV.")
    parser.add_argument(
        "-n", "--n-samples",
        type=int,
        required=True,
        help="Number of samples per generated CSV.",
    )
    parser.add_argument(
        "-s", "--n-sets",
        type=int,
        required=True,
        help="Number of sample CSV files to generate. Seeds 0..n_sets-1 are used.",
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        default=None,
        help="Output directory. Defaults to the parent directory of the input CSV.",
    )
    parser.add_argument(
        "-z", "--z-height",
        type=float,
        default=None,
        help="Optional z-height of the slice to sample from (not necessary for 2D input data).",
    )
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()

    wind_csv, rows = load_wind_rows(args.input_csv)
    rows = select_z_slice(rows, args.z_height)

    output_dir = wind_csv.parent if args.output_dir is None else Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    for seed in range(args.n_sets):
        sampled = sample_rows(rows, args.n_samples, seed)
        out_path = output_dir / f"sample_points_n{args.n_samples}_seed{seed}.csv"
        sampled.to_csv(out_path, index=False)
        if args.verbose:
            print(f"Saved sample set with {args.n_samples} samples to {out_path}")

if __name__ == "__main__":
    main()
