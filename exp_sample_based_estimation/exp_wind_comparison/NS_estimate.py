import gmsh
import ufl
import numpy as np
import matplotlib.pyplot as plt

from mpi4py import MPI
from dolfinx import fem
from pathlib import Path
from tools.airflow_estimator import AirflowEstimator
from tools.visualizer import MatplotlibVisualizer2D

# Example usage 
bpfile   = Path("/app/exp_sample_based_estimation/exp_wind_comparison/airflow_10x6_ground_truth.bp")
meshfile = Path("/app/exp_sample_based_estimation/exp_wind_comparison/mesh.msh")

est = AirflowEstimator.from_file(bpfile, fun_name="velocity_H2", meshtags_name="facet_tags", p=50, seed=1)
est.set_weights(kin_v=1.5e-4, misfit=1e0, pde_err=1e1, reg=1e-4)

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
est._boundary_name_to_id = name_to_id

# No slip boundary condition
u_D_no_slip = fem.Function(V)
u_D_no_slip.x.array[:] = 0.0
no_slip_names = ["Walls"]
dofs_local = np.concatenate([facet_tags.find(name_to_id[name]) for name in no_slip_names])
dofs_no_slip = fem.locate_dofs_topological((W0, V), V.mesh.topology.dim - 1, dofs_local)

bc_no_slip = fem.dirichletbc(u_D_no_slip, dofs_no_slip, W0)

# Outflow - zero pressure
p_zero = fem.Function(Q)
p_zero.x.array[:] = 0.0
p_1_facets = ["Outlet"]
dofs_local = np.concatenate([facet_tags.find(name_to_id[name]) for name in p_1_facets])
dofs_out = fem.locate_dofs_topological((W1, Q), V.mesh.topology.dim - 1, dofs_local)

bc_out = fem.dirichletbc(p_zero, dofs_out, W1)

est.add_dirichlet_bc([bc_no_slip, bc_out])

result = est.solve(maxit=3)
terms = est.evaluate_objective_terms(result)
print("Objective terms:")
for k, v in terms.items():
    print(f"  {k:>24s}: {v:.6e}")
u_est  = result.sub(0).collapse()

vis = MatplotlibVisualizer2D(u_est.function_space)
vis.add_background_mesh()
vis.add_streamplot("Estimated Airflow", u_est, 100, 50, 2.0)
vis.show("Estimated Airflow from LS-FEM", "streamplot_estimation.png")

vis2 = MatplotlibVisualizer2D(u_est.function_space)
vis2.add_background_mesh()
vis2.add_vector_field("Estimated Airflow", u_est)
vis2.add_points(est.get_measurement_coordinates())
vis2.show("Estimated Airflow from LS-FEM", "quiverplot_estimation.png")
