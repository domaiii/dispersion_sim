import numpy as np
from mpi4py import MPI
from basix.ufl import element
from dolfinx import mesh, fem, io, plot
from ufl import TrialFunction, TestFunction, dx

import pyvista
import gmsh
import ufl
from pathlib import Path
from dolfinx.fem.petsc import LinearProblem

from dolfinx import *

y_lim = 10.0
x_lim = 10.0

meshfile = Path("/home/dominik/git/dispersion_sim/meshes/square_holes_ad/mesh.msh").resolve()
domain, cell_tags, facet_tags = io.gmshio.read_from_msh(meshfile, MPI.COMM_WORLD, gdim=2)
V = fem.functionspace(domain, element("CG", domain.basix_cell(), 1))

# Parameters
D_phys = fem.Constant(domain, 2.1e-5)
theta = fem.Constant(domain, 1.0)
Pe = fem.Constant(domain, 1e2)
t_end = 10 
dt = 0.1

# Velocity field and boundary conditions
def velocity_field(x):
    return np.vstack((np.ones_like(x[0]), np.ones_like(x[1])))

V_vec = fem.functionspace(domain, ("CG", 1, (domain.geometry.dim,)))
beta = fem.Function(V_vec)
beta.interpolate(velocity_field)

gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 1)
gmsh.open(str(meshfile))

phy_groups = gmsh.model.getPhysicalGroups()
name_to_id = {
    gmsh.model.getPhysicalName(dim, tag): tag for dim, tag in phy_groups
}
print(name_to_id)

gmsh.finalize()

walls_tag = name_to_id["Walls"]
inflow_tag = name_to_id["Inflow"]

fdim = domain.topology.dim - 1

walls_dofs = fem.locate_dofs_topological(V, fdim, facet_tags.find(walls_tag))
u_zero = fem.Constant(domain, 0.0)
bc_zero = fem.dirichletbc(u_zero, walls_dofs, V)

inflow_dofs = fem.locate_dofs_topological(V, fdim, facet_tags.find(inflow_tag))
u_one = fem.Constant(domain, 1.0)
bc_inlet = fem.dirichletbc(u_one, inflow_dofs, V)


# PDE weak formulation + SUPG
U_char = ufl.sqrt(ufl.dot(beta, beta))
h = ufl.CellDiameter(domain)
# Alternative: Determine Peclet number automatically: 
# #Pe = U_char * h / D_phys, but are the vals correct?
nb = ufl.sqrt(ufl.inner(beta, beta))
tau = 0.5 * h * pow(4.0 / (Pe*h) + 2.0 * nb, -1.0)

u0 = fem.Function(V)
u0.x.array[:] = 0.0
u = TrialFunction(V)
v = TestFunction(V)

A1 = (1.0/Pe) * ufl.inner(ufl.grad(u), ufl.grad(v)) * dx + \
     ufl.inner(beta, ufl.grad(u)) * v * dx

A0 = (1.0/Pe) * ufl.inner(ufl.grad(u0), ufl.grad(v)) * dx + \
     ufl.inner(beta, ufl.grad(u0)) * v * dx

A = (1/dt)*ufl.inner(u, v) * dx - (1/dt)* ufl.inner(u0,v) * dx + theta* A1 + (1-theta) * A0

r=(((1/dt)*(u-u0) + theta*((1.0/Pe)* ufl.div(ufl.grad(u)) + ufl.inner(beta,ufl.grad(u))) + 
   (1-theta)*((1.0/Pe)*ufl.div(ufl.grad(u0)) + ufl.inner(beta,ufl.grad(u0))) ) * 
  tau * ufl.inner(beta,ufl.grad(v))*dx)

F = A + r

# Set up problem
problem = LinearProblem(ufl.lhs(F), ufl.rhs(F), bcs=[bc_zero, bc_inlet], 
                        petsc_options={"ksp_type": "gmres", "pc_type": "ilu"})
u = fem.Function(V)

xdmf = io.XDMFFile(domain.comm, "results_ad_with_obstacles.xdmf", "w")
xdmf.write_mesh(domain)

# Solve
t = 0.0
u.x.array[:] = u0.x.array

while t < t_end:
    print(f"t = {t:.2f} of {t_end:.2f} s")

    u = problem.solve()
    xdmf.write_function(u, t + dt)
    t += dt
    u0.x.array[:] = u.x.array

xdmf.close()

# domain.topology.create_entities(1)
# domain.topology.create_connectivity(1, 2)
# topology, cell_type, geometry = plot.vtk_mesh(domain, 1)
# grid = pyvista.UnstructuredGrid(topology, cell_type, geometry)
# grid.point_data["concentration"] = u.x.array.real
# plotter = pyvista.Plotter()
# plotter.add_mesh(grid, show_edges=True)
# plotter.show()