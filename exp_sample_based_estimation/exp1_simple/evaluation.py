import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
WDIR = Path(__file__).parent
DATAFILE = WDIR / "exp1_results.parquet"
OUTPUT_DIR = WDIR / "result_plots"
OUTPUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------
# Global style
# ---------------------------------------------------------
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "font.size": 13,
    "axes.labelsize": 14,
    "axes.titlesize": 15,
    "lines.linewidth": 2.2,
    "lines.markersize": 7,
    "legend.fontsize": 12,
    "axes.edgecolor": "0.3",
    "axes.titlepad": 12,
})

# ---------------------------------------------------------
# Legend configuration (centralized)
# ---------------------------------------------------------
LEGEND_KW = dict(
    loc="upper center",
    bbox_to_anchor=(0.5, -0.18),
    ncol=3,
    frameon=False,
    fontsize=12,
    title_fontsize=12
)

def add_legend(ax, title=None):
    leg = ax.legend(**LEGEND_KW)
    if title:
        leg.set_title(title)

# ---------------------------------------------------------
# Helper: save figure as vector PDF (+ optional show)
# ---------------------------------------------------------
def save_fig(fig, name: str, show=True):
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"{name}.pdf", format="pdf")
    if show:
        fig.show()

# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------
df = pd.read_parquet(DATAFILE)

MAX_P_GAS = 800
MAX_P_WIND = 800

p_wind_values = sorted([v for v in df["p_wind"].unique() if np.isfinite(v)])
p_gas_values  = sorted(df["p_gas"].unique())

p_gas_values  = [v for v in p_gas_values  if v <= MAX_P_GAS]
p_wind_values = [v for v in p_wind_values if v <= MAX_P_WIND]

grouped = df.groupby(["p_wind", "p_gas"]).mean(numeric_only=True).reset_index()

baseline = (
    df[df["wind_source"] == "true"]
    .groupby("p_gas")
    .mean(numeric_only=True)
    .reset_index()
)

wind_error = (
    df[df["wind_source"] == "reconstructed"]
    .groupby("p_wind")
    .mean(numeric_only=True)
    .reset_index()
)

# color palette
palette_grey = plt.cm.Greys(np.linspace(0.3, 0.8, len(p_wind_values)))
palette_pw = plt.cm.Blues(np.linspace(0.3, 1.0, len(p_wind_values)))
palette_pg = plt.cm.Oranges(np.linspace(0.3, 1.0, len(p_wind_values)))

# ---------------------------------------------------------
# (a) Localization Error vs Gas Samples
# ---------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 6))
for color, p_w in zip(palette_pw, p_wind_values):
    sub = grouped[(grouped["p_wind"] == p_w) & (grouped["p_gas"] <= MAX_P_GAS)]
    ax.plot(sub["p_gas"], sub["loc_error"], marker="o", color=color,
            label=f"{int(p_w)}")

b = baseline[baseline["p_gas"] <= MAX_P_GAS]
# ax.plot(b["p_gas"], b["loc_error"], "k--", linewidth=2.5, label="True wind")

ax.set_xlabel("Number of Gas Samples")
ax.set_ylabel("Localization Error (m)")
ax.set_title("Localization Error vs. Number of Gas Samples")
add_legend(ax, "Wind Sample Size")
save_fig(fig, "loc_error_vs_gas")

# ---------------------------------------------------------
# (b) Localization Error vs Wind Samples
# ---------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 6))
for color, p_g in zip(palette_pg, p_gas_values):
    sub = grouped[(grouped["p_gas"] == p_g) & (grouped["p_wind"] <= MAX_P_WIND)]
    ax.plot(sub["p_wind"], sub["loc_error"], marker="o", color=color,
            label=f"{int(p_g)}")

ax.set_xlabel("Number of Wind Samples")
ax.set_ylabel("Localization Error (m)")
ax.set_title("Localization Error vs. Number of Wind Samples")
add_legend(ax, "Gas Sample Size")
save_fig(fig, "loc_error_vs_wind")

# ---------------------------------------------------------
# (c) Plume Error vs Gas Samples
# ---------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 6))
for color, p_w in zip(palette_pw, p_wind_values):
    sub = grouped[(grouped["p_wind"] == p_w) & (grouped["p_gas"] <= MAX_P_GAS)]
    ax.plot(sub["p_gas"], sub["normalized_plume_err_L2"], marker="o", color=color,
            label=f"{int(p_w)}")

b = baseline[baseline["p_gas"] <= MAX_P_GAS]
# ax.plot(b["p_gas"], b["rel_plume_L2"], "k--", linewidth=2.5, label="True wind")

ax.set_xlabel("Number of Gas Samples")
ax.set_ylabel("Plume Error (RMS)")
ax.set_title("Plume Error (RMS) vs. Number of Gas Samples")
add_legend(ax, "Wind Sample Size")
save_fig(fig, "plume_vs_gas")

# ---------------------------------------------------------
# (d) Plume Error vs Wind Samples
# ---------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 6))
for color, p_g in zip(palette_pg, p_gas_values):
    sub = grouped[(grouped["p_gas"] == p_g) & (grouped["p_wind"] <= MAX_P_WIND)]
    ax.plot(sub["p_wind"], sub["normalized_plume_err_L2"], marker="o", color=color,
            label=f"{int(p_g)}")

ax.set_xlabel("Number of Wind Samples")
ax.set_ylabel("Plume Error (RMS)")
ax.set_title("Plume Error (RMS) vs. Number of Wind Samples")
add_legend(ax, "Gas Sample Size")
save_fig(fig, "plume_vs_wind")

# ---------------------------------------------------------
# (e) Relative Wind Reconstruction Error
# ---------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 6))
w = wind_error[wind_error["p_wind"] <= 1000]
ax.plot(w["p_wind"], w["rel_wind_L2"],
        marker="o", color=palette_grey[-1])

ax.set_xlabel("Number of Wind Samples")
ax.set_ylabel("Relative Wind Error (L2)")
ax.set_title("Relative Wind Reconstruction Error")
add_legend(ax)
save_fig(fig, "wind_error_vs_wind")

# ---------------------------------------------------------
# (f) Absolute Plume Error vs Gas Samples
# ---------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 6))
for color, p_w in zip(palette_pw, p_wind_values):
    sub = grouped[(grouped["p_wind"] == p_w) & (grouped["p_gas"] <= 1000)]
    ax.plot(sub["p_gas"], sub["plume_L2"], marker="o", color=color,
            label=f"{int(p_w)}")

b = baseline[baseline["p_gas"] <= 1000]
# ax.plot(b["p_gas"], b["plume_L2"],
#         "k--", linewidth=2.5, label="True wind")

ax.set_xlabel("Number of Gas Samples")
ax.set_ylabel("Absolute Plume Error (L2)")
ax.set_title("Absolute Plume Error vs. Number of Gas Samples")
add_legend(ax, "Wind Sample Size")
save_fig(fig, "abs_plume_vs_gas")

# ---------------------------------------------------------
# (g) Wind Component Errors L2
# ---------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
w = wind_error[wind_error["p_wind"] <= 1000]
ax1.plot(w["p_wind"], w["angular_wind_err_L2"],
        marker="o", color=palette_grey[-1])
ax1.set_xlabel("Number of Wind Samples")
ax1.set_ylabel("Angular Error (L2)")
ax1.set_title("Angular Error of Estimated Wind")
add_legend(ax1)

ax2.plot(w["p_wind"], w["magnitude_err_L2"],
        marker="o", color=palette_grey[-1])
ax2.set_xlabel("Number of Wind Samples")
ax2.set_ylabel("Magnitude Error (L2)")
ax2.set_title("Magnitude Error of Estimated Wind")
add_legend(ax2)

save_fig(fig, "component_error_wind")


# ---------------------------------------------------------
# (h) Wind Component Errors RMS
# ---------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
w = wind_error[wind_error["p_wind"] <= 1000]
ax1.plot(w["p_wind"], w["angular_wind_err_RMS"] * 180.0 / np.pi,
        marker="o", color=palette_grey[-1])
ax1.set_xlabel("Number of Wind Samples")
ax1.set_ylabel("Angular Error (RMS) in °")
ax1.set_title("Angular Error of Estimated Wind")
add_legend(ax1)

ax2.plot(w["p_wind"], w["magnitude_err_RMS"],
        marker="o", color=palette_grey[-1])
ax2.set_xlabel("Number of Wind Samples")
ax2.set_ylabel("Magnitude Error (RMS) in m/s")
ax2.set_title("Magnitude Error of Estimated Wind")
add_legend(ax2)

save_fig(fig, "component_error_wind_RMS")
