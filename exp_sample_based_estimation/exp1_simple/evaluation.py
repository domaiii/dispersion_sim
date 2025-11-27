import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

df = pd.read_parquet("/home/dominik/git/dispersion_sim/exp_sample_based_estimation/exp1_simple/test_bug_exp.parquet")

# ---------------------------------------------------------
# Preprocess
# ---------------------------------------------------------
# Sorted unique values
p_wind_values = sorted([v for v in df["p_wind"].unique() if np.isfinite(v)])
p_gas_values  = sorted(df["p_gas"].unique())

# Mean over seeds & source locations
grouped = df.groupby(["p_wind", "p_gas"]).mean(numeric_only=True).reset_index()

# Separate baseline (true wind)
baseline = df[df["wind_source"] == "true"].groupby("p_gas").mean(numeric_only=True).reset_index()

# ---------------------------------------------------------
# Build figure
# ---------------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(22, 12))
axes = axes.flatten()

# =========================================================
# 1) Localization error vs gas samples (per wind samples)
# =========================================================
ax = axes[0]
for p_w in p_wind_values:
    sub = grouped[grouped["p_wind"] == p_w]
    ax.plot(sub["p_gas"], sub["loc_error"], marker="o", label=f"pw={p_w}")

# Add baseline (true wind)
ax.plot(baseline["p_gas"], baseline["loc_error"], "k--", linewidth=2.5, label="true wind")

ax.set_xlabel("Gas samples p_gas")
ax.set_ylabel("Localization error")
ax.set_title("Localization error vs gas samples")
ax.grid(True)
ax.legend()


# =========================================================
# 2) Localization error vs wind samples (per gas samples)
# =========================================================
ax = axes[1]
for p_g in p_gas_values:
    sub = grouped[grouped["p_gas"] == p_g]
    ax.plot(sub["p_wind"], sub["loc_error"], marker="o", label=f"pg={p_g}")

ax.set_xlabel("Wind samples p_wind")
ax.set_ylabel("Localization error")
ax.set_title("Localization error vs wind samples")
ax.grid(True)
ax.legend()


# =========================================================
# 3) Plume error vs gas samples (per wind samples)
# =========================================================
ax = axes[2]
for p_w in p_wind_values:
    sub = grouped[grouped["p_wind"] == p_w]
    ax.plot(sub["p_gas"], sub["rel_plume_L2"], marker="o", label=f"pw={p_w}")

# Baseline with true wind
ax.plot(baseline["p_gas"], baseline["rel_plume_L2"], "k--", linewidth=2.5, label="true wind")

ax.set_xlabel("Gas samples p_gas")
ax.set_ylabel("Relative plume L2 error")
ax.set_title("Relative Plume Error for Gas Samples")
ax.grid(True)
ax.legend()


# =========================================================
# 4) Plume error vs wind samples (per gas samples)
# =========================================================
ax = axes[3]
for p_g in p_gas_values:
    sub = grouped[grouped["p_gas"] == p_g]
    ax.plot(sub["p_wind"], sub["rel_plume_L2"], marker="o", label=f"pg={p_g}")

ax.set_xlabel("Wind samples p_wind")
ax.set_ylabel("Relative plume L2 error")
ax.set_title("Relative plume error vs wind samples")
ax.grid(True)
ax.legend()


# =========================================================
# 5) Wind L2 error vs wind samples
# =========================================================
ax = axes[4]

sub = df[df["wind_source"] == "reconstructed"]
wind_error = sub.groupby("p_wind").mean(numeric_only=True).reset_index()

ax.plot(wind_error["p_wind"], wind_error["rel_wind_L2"], marker="o")
ax.set_xlabel("Wind samples p_wind")
ax.set_ylabel("Relative wind L2 error")
ax.set_title("Relative Wind Reconstruction Error vs Wind Samples")
ax.grid(True)

# =========================================================
# 3) Plume error vs gas samples (per wind samples)
# =========================================================
ax = axes[5]
for p_w in p_wind_values:
    sub = grouped[grouped["p_wind"] == p_w]
    ax.plot(sub["p_gas"], sub["plume_L2"], marker="o", label=f"pw={p_w}")

# Baseline with true wind
ax.plot(baseline["p_gas"], baseline["plume_L2"], "k--", linewidth=2.5, label="true wind")

ax.set_xlabel("Gas samples p_gas")
ax.set_ylabel("Absolute plume L2 error")
ax.set_title("Absolute Plume Error for Gas Samples")
ax.grid(True)
ax.legend()
plt.tight_layout()
plt.show()