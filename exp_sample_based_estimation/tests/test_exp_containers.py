from pathlib import Path
import numpy as np
import gmsh

from dolfinx import fem
from tools.airflow_estimator import AirflowEstimator
from tools.gas_estimator import GasSourceEstimator
from tools.visualizer import Visualizer2D
from tools.experiment import SingleExperiment, SingleExperimentResult


# -------------------------------------------------------------
# 1. Load mesh + wind ground truth
# -------------------------------------------------------------

WDIR = Path(__file__).parent.resolve()

bpfile   = WDIR / "airflow.bp"
meshfile = WDIR / "mesh/mesh.msh"

air_new = AirflowEstimator.from_file(
    bpfile,
    p=100,
    seed=42,
    fun_name="velocity_H2",
    meshtags_name="facet_tags",
    meshfile=meshfile
)

air_new.set_weights(
    kin_v=1.5e-4,
    misfit=1.0,
    pde_err=1.0,
    reg=1e-3
)

air_new.set_zero_pressure_bc("Outflow")
air_new.set_no_slip_bc(["Walls", "Obstacles"])

# -------------------------------------------------------------
# 2. Gas estimator (empty)
# -------------------------------------------------------------

gas_new = GasSourceEstimator(air_new.domain)

# -------------------------------------------------------------
# 3. Run experiment
# -------------------------------------------------------------

exp = SingleExperiment(
    air_est=air_new,
    gas_est=gas_new,
    source_location=(0.7 * gas_new.Lx, 0.8 * gas_new.Ly),
    sigma_factor=0.01,
    p_wind=100,
    p_gas=60,
    wind_seed=42,
    gas_seed=15,
    gamma_reg=5e-2,
    amplitude=10.0
)

result = exp.run_L1()
result2 = exp.run_L2()

# -------------------------------------------------------------
# 4. Visualization (L1 result)
# -------------------------------------------------------------

vis = Visualizer2D(gas_new.scalar_space)
vis.add_scalar_field("f_true", result.f_true)
vis.add_scalar_field("f_est", result.f_est)
vis.add_points([result.true_location], color="blue", size=15, label="True Source Max")
vis.add_points([result.est_location], color="red", size=15, label="Estimated Source Max")
vis.show("Gas Source: True vs Estimated")

# Gas plume
vis2 = Visualizer2D(gas_new.scalar_space, font_size=26)
vis2.add_scalar_field("GT plume", result.c_true)
vis2.add_points(result.gas_sample_coords, color="orange", size=12, label="Gas Measurements")
vis2.show("Measurement Positions")

# Wind estimation
vis3 = Visualizer2D(air_new.V, font_size=26)
vis3.add_background_mesh()
vis3.add_points(result.wind_sample_coords, color="orange", size=12, label="Wind Measurements")
vis3.add_vector_field("Estimated Wind", result.u_est)
vis3.show("Airflow Estimation")

