from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


SINGLE_LAYER_Z_SPAN = 0.1


def _optional_path(root: Path, value: str | None) -> Path | None:
    return None if value is None else (root / value).resolve()


@dataclass(frozen=True)
class ScenarioConfig:
    name: str
    root: Path
    wind_csv: Path
    sample_dir: Path
    result_dir: Path
    wind_noise_std: float = 0.0
    z_height: float | None = None
    z_tol: float = 0.05
    max_xy_dist: float = 0.2
    wind_sample_sizes: list[int] = field(default_factory=list)
    mesh: Path | None = None
    occupancy_yaml: Path | None = None
    occupancy_image: Path | None = None
    wall_pattern: str = r"wall|obstacle"
    outflow_pattern: str = r"outlet|outflow"
    solver: str = "minimum_residual"
    regularization: str = "smooth"
    maxit: int = 25
    tol: float = 1e-2
    damping: float | None = None
    viscosity: float = 1e-5
    weight_misfit: float = 1e2
    weight_pde_res: float = 1.0
    weight_reg: float = 1e-2
    weight_boundary: float = 1e4
    gmrf_cell_size: float = 0.25
    variance_speed: float = 0.01
    variance_direction: float = 0.01
    batch_size: int = 50
    gp_length_scale: float = 1.0
    gp_optimize_length_scale: bool = True

    @classmethod
    def load(cls, scenario: str | Path) -> "ScenarioConfig":
        scenario = Path(scenario).resolve()
        if scenario.is_dir():
            scenario = scenario / "scenario.yaml"

        with scenario.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        root = scenario.parent
        geometry = raw.get("geometry", {})
        data = raw.get("data", {})
        noise = raw.get("measurement_noise", {})
        slicing = raw.get("ground_truth_slicing", {})
        solver = raw.get("ns_solver_parameters", {})
        gmrf = raw.get("gmrf_parameters", {})
        gp = raw.get("gp_parameters", {})
        damping = solver.get("damping", cls.damping)

        return cls(
            name=str(raw["name"]),
            root=root,
            mesh=_optional_path(root, geometry.get("mesh")),
            occupancy_yaml=_optional_path(root, geometry.get("occupancy_yaml")),
            occupancy_image=_optional_path(root, geometry.get("occupancy_image")),
            wall_pattern=str(geometry.get("wall_pattern", cls.wall_pattern)),
            outflow_pattern=str(geometry.get("outflow_pattern", cls.outflow_pattern)),
            wind_csv=(root / data["wind_csv"]).resolve(),
            sample_dir=(root / data["sample_dir"]).resolve(),
            result_dir=(root / data["result_dir"]).resolve(),
            wind_noise_std=(float(noise.get("wind_noise_std", cls.wind_noise_std))),
            z_height=None if slicing.get("z_height") is None else float(slicing["z_height"]),
            z_tol=float(slicing.get("z_tol", cls.z_tol)),
            max_xy_dist=float(slicing.get("max_xy_dist", cls.max_xy_dist)),
            wind_sample_sizes=[int(size) for size in raw.get("wind_sample_sizes", [])],
            solver=str(solver.get("solver", cls.solver)),
            regularization=str(solver.get("regularization", cls.regularization)),
            maxit=int(solver.get("maxit", cls.maxit)),
            tol=float(solver.get("tol", cls.tol)),
            damping=None if damping is None else float(damping),
            viscosity=float(solver.get("viscosity", cls.viscosity)),
            weight_misfit=float(solver.get("weight_misfit", cls.weight_misfit)),
            weight_pde_res=float(solver.get("weight_pde_res", cls.weight_pde_res)),
            weight_reg=float(solver.get("weight_reg", cls.weight_reg)),
            weight_boundary=float(solver.get("weight_boundary", cls.weight_boundary)),
            gmrf_cell_size=float(gmrf.get("cell_size", cls.gmrf_cell_size)),
            variance_speed=float(gmrf.get("var_speed", cls.variance_speed)),
            variance_direction=float(gmrf.get("var_direction", cls.variance_direction)),
            batch_size=int(gmrf.get("batch_size", cls.batch_size)),
            gp_length_scale=float(gp.get("length_scale", cls.gp_length_scale)),
            gp_optimize_length_scale=bool(gp.get("optimize_length_scale", cls.gp_optimize_length_scale))
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
