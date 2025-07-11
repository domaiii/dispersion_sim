import ufl
import pyvista
import numpy as np

from mpi4py import MPI
from dolfinx.fem import Function
from dolfinx import mesh, fem
from dolfinx.plot import vtk_mesh
from dolfinx.fem.petsc import LinearProblem
from basix.ufl import element
from petsc4py import PETSc

from ns_helpers import (create_2d_mesh_rect, 
                        create_noslip_dirichlet_bc, 
                        constant_pressure_bc, 
                        velocity_profile_bc)

# Mesh
x_lim = 200.0
y_lim = 100.0
x_res = 100
y_res = 50

domain = create_2d_mesh_rect(x_lim, y_lim, x_res, y_res, cell_type=mesh.CellType.triangle)

# Spaces
V = fem.functionspace(domain, element("Lagrange", domain.basix_cell(), 2, shape=(2,)))
Q = fem.functionspace(domain, element("Lagrange", domain.basix_cell(), 1))

# Physical parameters
t = 0.0
T = 1.0
num_steps = 500
dt = T / num_steps

nu = 1.0
rho = 1.0

# Trial and test functions
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
p = ufl.TrialFunction(Q)
q = ufl.TestFunction(Q)

# Functions
u_n = Function(V)
u_tent = Function(V)
u_new = Function(V)
p_n = Function(Q)
p_new = Function(Q)
f = fem.Constant(domain, PETSc.ScalarType((0.0, 0.0)))

def no_slip_bdry(x: np.ndarray) -> np.ndarray:
    top_bottom =  np.logical_or(np.isclose(x[1], y_lim), np.isclose(x[1], 0.0))
    lower_left = np.logical_and(np.isclose(x[0], 0.0), x[1] < y_lim / 2.0)
    upper_right = np.logical_and(np.isclose(x[0], x_lim), x[1] > y_lim / 2.0)
    side_no_slip = np.logical_or(lower_left, upper_right)

    return np.logical_or(side_no_slip, top_bottom)

def inflow(x: np.ndarray) -> np.ndarray:
    return np.logical_and(np.isclose(x[0], 0.0), x[1] > y_lim / 2.0)

def outflow(x: np.ndarray) -> np.ndarray:
    return np.logical_and(np.isclose(x[0], x_lim), x[1] < y_lim / 2.0)

def inflow_velocity_profile(x: np.ndarray) -> np.ndarray:
    y_centered = (x[1] - y_lim/2) / (y_lim/2)
    u_val = np.where(
        inflow(x),
        -4.0 * (1 - y_centered) * y_centered,
        0.0
    )
    return np.stack((u_val, np.zeros_like(x[1])))

bc_no_slip = create_noslip_dirichlet_bc(V, no_slip_bdry)
bc_in = velocity_profile_bc(V, inflow, inflow_velocity_profile)
bc_out = constant_pressure_bc(Q, outflow)

# Step 1: Tentative velocity without pressure gradient
a1 = ((1/dt)*ufl.inner(u, v)*ufl.dx 
      + nu*ufl.inner(ufl.grad(u), ufl.grad(v))*ufl.dx)
L1 = ((1/dt)*ufl.inner(u_n, v)*ufl.dx 
      - ufl.inner(ufl.dot(u_n, ufl.nabla_grad(u_n)), v)*ufl.dx 
      + ufl.inner(f, v)*ufl.dx)

tentative_velocity_problem = LinearProblem(a1, L1, [bc_no_slip, bc_in], u_tent, 
                                           petsc_options={"ksp_type": "gmres"})

# Step 2: Pressure correction from poisson equation
a2 = ufl.dot(ufl.grad(p), ufl.grad(q)) * ufl.dx
L2 = (rho/dt)*ufl.div(u_tent)*q*ufl.dx

pressure_problem = LinearProblem(a2, L2, [bc_out], p_new)

# Step 3: Velocity correction
a3 = ufl.inner(u, v)*ufl.dx
L3 = ufl.inner(u_tent, v)*ufl.dx - dt/rho * ufl.inner(ufl.grad(p_new), v)*ufl.dx

velocity_correction_problem = LinearProblem(a3, L3, [bc_no_slip, bc_in], u_new)


while t < T:
    tentative_velocity_problem.solve()
    pressure_problem.solve()
    velocity_correction_problem.solve()

    u_n.x.array[:] = u_new.x.array
    p_n.x.array[:] = p_new.x.array

    print(f"{t:.2f} s: Step succeeded")
    t += dt


topology, cell_types, geometry = vtk_mesh(V)
grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)

u_vals = u_n.x.array.reshape((-1, 2))  
u_full = np.zeros((u_vals.shape[0], 3))
u_full[:, :2] = u_vals               

u_magnitude = np.linalg.norm(u_vals, axis=1)

grid["u"] = u_full
grid["u_magnitude"] = u_magnitude

glyphs = grid.glyph(
    orient="u",
    scale="u_magnitude",  
    factor=0.1
)

plotter = pyvista.Plotter()
plotter.add_mesh(grid, style="wireframe", color="gray")
plotter.add_mesh(glyphs, scalars="u_magnitude", cmap="coolwarm")
plotter.add_axes()
plotter.view_xy()
plotter.show()



