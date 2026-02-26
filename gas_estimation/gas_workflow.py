import numpy as np
import adios4dolfinx
from pathlib import Path
from dolfinx import mesh, fem, io
from mpi4py import MPI
from basix.ufl import element
from tools.visualizer import Visualizer2D
from tools.gas_estimator import GasSourceEstimator

wind_file = Path("/app/wind_data/airflow_ipcs.bp")
domain = adios4dolfinx.read_mesh(wind_file, MPI.COMM_WORLD)
V = fem.functionspace(domain, element("Lagrange", domain.basix_cell(), 2, shape=(2,)))
wind_field = fem.Function(V)
adios4dolfinx.read_function(wind_file, wind_field, name="velocity_H2")

gas_est = GasSourceEstimator(domain, wind_field)

# Define ground truth
x0, y0 = 0.5 * gas_est.Lx, 0.5 * gas_est.Ly
sigma_x, sigma_y = 0.01 * gas_est.Lx, 0.01 * gas_est.Ly

def source_term(x: np.ndarray):
    return 10.0 * np.exp(-((x[0] - x0)**2 / (2 * sigma_x**2) + (x[1] - y0)**2 / (2 * sigma_y**2)))
f_true = fem.Function(gas_est.scalar_space)
f_true.interpolate(source_term)

gas_est.set_ground_truth(f_true)

# Generate random measurements
gas_est.reset_random_measurements(100, seed=5)

# Estimate source
gas_est.solve_L1(gamma_reg=0.001)


vis = Visualizer2D(gas_est.scalar_space, font_size=30)

vis.add_scalar_field("source_estimate", gas_est.source_est)

vis.add_points(gas_est.scalar_space.tabulate_dof_coordinates()[gas_est.m_ids], 
               color="orange", size=10, label="Measurements")

vis.show("Reconstruction")
