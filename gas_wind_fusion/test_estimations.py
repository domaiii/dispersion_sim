from pathlib import Path
import numpy as np
import gmsh
from mpi4py import MPI

from dolfinx import fem
from tools.airflow_estimator import AirflowEstimator, Visualizer2D
from tools.gas_estimator import GasSourceEstimator


# 1. Wind-Mesh & Ground-Truth-Wind laden + AirflowEstimator bauen
bpfile   = Path("/home/dominik/git/dispersion_sim/wind_data/airflow_ipcs.bp")
meshfile = Path("/home/dominik/git/dispersion_sim/meshes/rectangle_circular_holes/mesh.msh")

air = AirflowEstimator.from_file(
    bpfile,
    fun_name="velocity_H2",
    meshtags_name="facet_tags",
    p=100,
    seed=15
)

air.set_weights(kin_v=1.5e-4, misfit=1.0, pde_err=1.0, reg=1e-3)

# 2. Boundary Conditions setzen
gmsh.initialize()
gmsh.open(str(meshfile))
phy_groups = gmsh.model.getPhysicalGroups()
name_to_id = {gmsh.model.getPhysicalName(dim, tag): tag for (dim, tag) in phy_groups}
gmsh.finalize()

facet_tags = air.facet_tags
domain = air.domain

# No-slip BC
u_D_no_slip = fem.Function(air.V)
u_D_no_slip.x.array[:] = 0.0
walls = ["Walls", "Obstacle1", "Obstacle2", "Obstacle3", "Obstacle4", "Obstacle5"]
wall_facets = np.concatenate([facet_tags.find(name_to_id[name]) for name in walls])
dofs_wall = fem.locate_dofs_topological((air.W0, air.V), domain.topology.dim-1, wall_facets)
bc_no_slip = fem.dirichletbc(u_D_no_slip, dofs_wall, air.W0)

# Outflow p=0
p_zero = fem.Function(air.Q)
p_zero.x.array[:] = 0.0
outflow_facets = facet_tags.find(name_to_id["Outflow"])
dofs_out = fem.locate_dofs_topological((air.W1, air.Q), domain.topology.dim-1, outflow_facets)
bc_out = fem.dirichletbc(p_zero, dofs_out, air.W1)

air.add_dirichlet_bc([bc_no_slip, bc_out])


# 3. Windmessungen erzeugen + Wind schätzen
air.reset_random_measurements(p=100, seed=42)
u_est = air.solve(maxit=5).sub(0).collapse()


# 4. GasEstimator erzeugen basierend auf geschätztem Wind
gas = GasSourceEstimator(domain, u_est)


# 5. Gas-Quelle (Ground Truth) definieren
x0, y0 = 0.7 * gas.Lx, 0.8 * gas.Ly
sigma_x, sigma_y = 0.01 * gas.Lx, 0.01 * gas.Ly

def source_term(x):
    return 10.0 * np.exp(
        -((x[0] - x0)**2 / (2*sigma_x**2) +
          (x[1] - y0)**2 / (2*sigma_y**2))
    )

f_true = fem.Function(gas.scalar_space)
f_true.interpolate(source_term)

gas.set_ground_truth(f_true)

# 6. Gasmessungen erzeugen
gas.generate_measurements_from_ground_truth(p=60, seed=15)


# 7. Gasquelle schätzen (L1 oder L2)
f_est = gas.solve_L1(gamma_reg=5e-2)

print("Estimated source center:", gas.estimated_source_max_location())
print("Estimated max source value:", gas.estimated_source_max_value())

# Visualizer erzeugen
vis = Visualizer2D(gas.scalar_space)

# Hintergrund-Mesh (optional)
vis.add_background_mesh(opacity=0.1)

# Ground truth source
vis.add_scalar_field("f_true", gas.f_true)

# Estimated source
vis.add_scalar_field("f_est", f_est)

# Maxima visualisieren
coords = gas.scalar_space.tabulate_dof_coordinates()
idx_true = np.argmax(gas.f_true.x.array)
loc_true = coords[idx_true]

idx_est = np.argmax(f_est.x.array)
loc_est = coords[idx_est]

# Punkte setzen
vis.add_points([loc_true], color="blue", size=15, label="True Source Max")
vis.add_points([loc_est], color="red", size=15, label="Estimated Source Max")

# Zeigen
vis.show(title="Gas Source: True vs Estimated", zoom=1.2)

# --- Plot B: Messpunkte + geschätzte Quelle ---
vis2 = Visualizer2D(gas.scalar_space, font_size=26)

# Estimated source field
vis2.add_scalar_field("Gas Distribution", gas.get_ground_truth_concentration(), cmap="viridis")

# Measurement point coordinates
meas_coords = gas.scalar_space.tabulate_dof_coordinates()[gas.m_ids]

vis2.add_points(meas_coords, 
                color="orange",
                size=12,
                label="Gas Measurements")

vis2.show("Measurement Positions")


# --- Plot C: Estimated Airflow ---
vis3 = Visualizer2D(air.V, font_size=26)

# Estimated source field
vis3.add_background_mesh()

# Measurement point coordinates
meas_coords = gas.scalar_space.tabulate_dof_coordinates()[gas.m_ids]

vis3.add_points(air.get_measurement_coordinates(), 
                color="orange",
                size=12,
                label="Wind Measurements")
vis3.add_vector_field("Estimated Wind", u_est, factor=1.0)

vis3.show("Airflow Estimation")

