
from pathlib import Path
import numpy as np
import gmsh
from mpi4py import MPI

from dolfinx import fem
from tools.airflow_estimator import AirflowEstimator
from tools.visualizer import Visualizer2D
from tools.gas_estimator import GasSourceEstimator

WDIR = Path(__file__).parent.resolve()

# 1. Wind-Mesh & Ground-Truth-Wind laden + AirflowEstimator bauen
bpfile = WDIR / "airflow.bp"
meshfile = WDIR / "mesh/mesh.msh"

air_old = AirflowEstimator.from_bp(
    bpfile,
    fun_name="velocity_H2",
    meshtags_name="facet_tags",
    p=100,
    seed=15
)

air_old.set_weights(kin_v=1.5e-4, misfit=1.0, pde_err=1.0, reg=1e-3)

# 2. Boundary Conditions setzen
gmsh.initialize()
gmsh.open(str(meshfile))
phy_groups = gmsh.model.getPhysicalGroups()
name_to_id = {gmsh.model.getPhysicalName(dim, tag): tag for (dim, tag) in phy_groups}
gmsh.finalize()

facet_tags = air_old.facet_tags
domain = air_old.domain

# No-slip BC
u_D_no_slip = fem.Function(air_old.V)
u_D_no_slip.x.array[:] = 0.0
walls = ["Walls", "Obstacles"]
wall_facets = np.concatenate([facet_tags.find(name_to_id[name]) for name in walls])
dofs_wall = fem.locate_dofs_topological((air_old.W0, air_old.V), domain.topology.dim-1, wall_facets)
bc_no_slip = fem.dirichletbc(u_D_no_slip, dofs_wall, air_old.W0)

# Outflow p=0
p_zero = fem.Function(air_old.Q)
p_zero.x.array[:] = 0.0
outflow_facets = facet_tags.find(name_to_id["Outflow"])
dofs_out = fem.locate_dofs_topological((air_old.W1, air_old.Q), domain.topology.dim-1, outflow_facets)
bc_out = fem.dirichletbc(p_zero, dofs_out, air_old.W1)

air_old.add_dirichlet_bc([bc_no_slip, bc_out])


# 3. Windmessungen erzeugen + Wind schätzen
air_old.reset_random_measurements(p=100, seed=42)
u_est = air_old.solve_minimum_residual(maxit=5).sub(0).collapse()


# 4. GasEstimator erzeugen basierend auf geschätztem Wind
gas_old = GasSourceEstimator(domain, u_est)


# 5. Gas-Quelle (Ground Truth) definieren
x0, y0 = 0.7 * gas_old.Lx, 0.8 * gas_old.Ly
sigma_x = 0.01 * gas_old.Lx

def source_term(x):
    return 10.0 * np.exp(
        -((x[0] - x0)**2 / (2*sigma_x**2) +
          (x[1] - y0)**2 / (2*sigma_x**2))
    )

f_true = fem.Function(gas_old.scalar_space)
f_true.interpolate(source_term)

gas_old.set_ground_truth(f_true)

# 6. Gasmessungen erzeugen
gas_old.reset_random_measurements(p=60, seed=15)


# 7. Gasquelle schätzen (L1 oder L2)
f_est = gas_old.solve_L1(gamma_reg=5e-2)


# Visualizer erzeugen
vis = Visualizer2D(gas_old.scalar_space)

# Hintergrund-Mesh (optional)
# vis.add_background_mesh(opacity=0.1)

# Ground truth source
vis.add_scalar_field("f_true", gas_old.f_true)

# Estimated source
vis.add_scalar_field("f_est", f_est)

# Maxima visualisieren
coords = gas_old.scalar_space.tabulate_dof_coordinates()
idx_true = np.argmax(gas_old.f_true.x.array)
loc_true = coords[idx_true]

idx_est = np.argmax(f_est.x.array)
loc_est = coords[idx_est]

# Punkte setzen
vis.add_points([loc_true], color="blue", size=15, label="True Source Max")
vis.add_points([loc_est], color="red", size=15, label="Estimated Source Max")

# Zeigen
vis.show(title="Gas Source: True vs Estimated", zoom=1.2)

# --- Plot B: Messpunkte + geschätzte Quelle ---
vis2 = Visualizer2D(gas_old.scalar_space, font_size=26)

# Estimated source field
vis2.add_scalar_field("Gas Distribution", gas_old.dispersion_for_true_source())

# Measurement point coordinates
meas_coords = gas_old.scalar_space.tabulate_dof_coordinates()[gas_old.m_ids]

vis2.add_points(meas_coords, 
                color="orange",
                size=12,
                label="Gas Measurements")

vis2.show("Measurement Positions")


# --- Plot C: Estimated Airflow ---
vis3 = Visualizer2D(air_old.V, font_size=26)

# Estimated source field
vis3.add_background_mesh()

# Measurement point coordinates
meas_coords = gas_old.scalar_space.tabulate_dof_coordinates()[gas_old.m_ids]

vis3.add_points(air_old.get_measurement_coordinates(), 
                color="orange",
                size=12,
                label="Wind Measurements")
vis3.add_vector_field("Estimated Wind", u_est, factor=1.0)

vis3.show("Airflow Estimation")
