import gmsh
import ufl
import numpy as np

from mpi4py import MPI
from dolfinx import fem
from pathlib import Path
from tools.airflow_estimator import AirflowEstimator
from tools.visualizer import MatplotlibVisualizer2D

### Example usage of the AirflowEstimator

# Load ground truth wind and mesh file
bpfile   = Path("/app/exp_sample_based_estimation/exp1_simple/airflow_picard.bp")
meshfile = Path("/app/exp_sample_based_estimation/exp1_simple/mesh/mesh.msh")

est = AirflowEstimator.from_file(bpfile, fun_name="velocity_H2", meshtags_name="facet_tags", p=100, seed=15)
est.set_weights(kin_v=1.5e-4, misfit=1e2, pde_err=1.0, reg=1e-3)

# Boundary conditions
V = est.V
Q = est.Q
W0 = est.W0
W1 = est.W1
W = est.W

gmsh.initialize()
gmsh.open(str(meshfile))
phy_groups = gmsh.model.getPhysicalGroups()
name_to_id = {gmsh.model.getPhysicalName(dim, tag): tag for (dim, tag) in phy_groups}
gmsh.finalize()

facet_tags = est.facet_tags

# No slip boundary condition
u_D_no_slip = fem.Function(V)
u_D_no_slip.x.array[:] = 0.0
no_slip_names = ["Walls", "Obstacles"]
dofs_local = np.concatenate([facet_tags.find(name_to_id[name]) for name in no_slip_names])
dofs_no_slip = fem.locate_dofs_topological((W0, V), V.mesh.topology.dim - 1, dofs_local)

bc_no_slip = fem.dirichletbc(u_D_no_slip, dofs_no_slip, W0)

# Outflow - zero pressure
p_zero = fem.Function(Q)
p_zero.x.array[:] = 0.0
p_1_facets = ["Outflow"]
dofs_local = np.concatenate([facet_tags.find(name_to_id[name]) for name in p_1_facets])
dofs_out = fem.locate_dofs_topological((W1, Q), V.mesh.topology.dim - 1, dofs_local)

bc_out = fem.dirichletbc(p_zero, dofs_out, W1)

est.add_dirichlet_bc([bc_no_slip, bc_out])

result = est.solve(maxit=3)

m_coords = est.get_measurement_coordinates()

vis = MatplotlibVisualizer2D(V)
vis.add_background_mesh()
vis.add_vector_field("Measurements from ground truth", est.ground_truth.sub(0).collapse())
vis.add_points(m_coords, "red", size=30, label="Measurements")
vis.show("Ground truth and measurement points" ,"gt_meas.png")

vis2 = MatplotlibVisualizer2D(V)
vis2.add_background_mesh()
vis2.add_vector_field("Airflow Estimation", result.sub(0).collapse())
vis2.show("Airflow Estimation", "estim.png")

# Compare result to ground truth
u_true = est.ground_truth.sub(0).collapse()
u_est  = result.sub(0).collapse()

u_true_arr = u_true.x.array.reshape(-1, 2)
u_est_arr  = u_est.x.array.reshape(-1, 2)

norm_true = np.linalg.norm(u_true_arr, axis=1)
norm_est  = np.linalg.norm(u_est_arr, axis=1)
eps = 1e-12
u_true_unit = u_true_arr / (norm_true[:, None] + eps)
u_est_unit  = u_est_arr  / (norm_est[:, None] + eps)

# Angular difference in degrees
dot = np.clip(np.sum(u_true_unit * u_est_unit, axis=1), -1.0, 1.0)
ang_diff = np.degrees(np.arccos(dot))

# Velocity magnitude difference
vel_diff = np.abs(norm_est - norm_true)

# Convert to Fenicsx scalar field
V_scalar = V.sub(0).collapse()[0]
V = u_true.function_space
ang_field = fem.Function(V_scalar)
vel_field = fem.Function(V_scalar)

ang_field.x.array[:] = ang_diff
vel_field.x.array[:] = vel_diff

vis_Scalar = MatplotlibVisualizer2D(V_scalar)
vis_Scalar.add_background_mesh()
vis_Scalar.add_scalar_field("Angular error in °", ang_field)
vis_Scalar.show("Angular error in °", "ang_err-png")

vis_Scalar2 = MatplotlibVisualizer2D(V_scalar)
vis_Scalar2.add_background_mesh()
vis_Scalar2.add_scalar_field("Velocity error in m/s", vel_field)
vis_Scalar2.show("Velocity error in m/s", "vel_error.png")
