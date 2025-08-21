import numpy as np
from mpi4py import MPI
from basix.ufl import element
from dolfinx import mesh, fem, io, plot
from ufl import TrialFunction, TestFunction, dx

from pylab import meshgrid
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import pyvista
import ufl
from pathlib import Path
from dolfinx.fem.petsc import LinearProblem

from dolfinx import *

# Create mesh and define function space
y_lim = 25.0
x_lim = 25.0

domain = mesh.create_rectangle(MPI.COMM_WORLD, ((0,0),(x_lim, y_lim)), [128, 128])
V = fem.functionspace(domain, element("CG", domain.basix_cell(), 1))

# Parameters
D_phys = fem.Constant(domain, 1.0)
theta = fem.Constant(domain, 0.5)
#Pe = fem.Constant(domain, 1e3)
t_end = 20
dt = 0.2

def velocity_field(x):
    return np.vstack((1.0 * np.ones_like(x[0]), 0.5 * np.ones_like(x[0])))

V_vec = fem.functionspace(domain, ("CG", 1, (domain.geometry.dim,)))
beta = fem.Function(V_vec)
beta.interpolate(velocity_field)

# Source term
def source_term(x: np.ndarray):
    values = np.zeros(x.shape[1])
    inside = np.where((x[0] < 0.21 * x_lim) & (x[0] > 0.19 * x_lim) & (x[1] < 0.61 * y_lim) & (x[1] > 0.59 * y_lim))
    values[inside] = 10.0 
    return values

f = fem.Function(V)
f.interpolate(source_term)

u0 = fem.Function(V)
u0.x.array[:] = 0.0

inlet_y_start = y_lim / 2 - 0.05 * y_lim
inlet_y_end = y_lim / 2 + 0.05 * y_lim

u_inlet = fem.Constant(domain, 1.0)

def boundary(x):
    return np.isclose(x[0], 0.0) | np.isclose(x[0], x_lim) | \
           np.isclose(x[1], 0.0) | np.isclose(x[1], y_lim)
    

u_zero = fem.Constant(domain, 0.0)
bc_zero = fem.dirichletbc(u_zero, fem.locate_dofs_geometrical(V, boundary), V)

U_char = ufl.sqrt(ufl.dot(beta, beta))
L_char = fem.Constant(domain, x_lim)
h = ufl.CellDiameter(domain)
Pe = U_char * L_char / D_phys
nb = ufl.sqrt(ufl.inner(beta, beta))

tau = 0.5 * h * pow(4.0 / (Pe*h) + 2.0 * nb, -1.0)

v = TestFunction(V)
u = TrialFunction(V)

A1 = (1.0/Pe) * ufl.inner(ufl.grad(u), ufl.grad(v)) * dx + \
     ufl.inner(beta, ufl.grad(u)) * v * dx

A0 = (1.0/Pe) * ufl.inner(ufl.grad(u0), ufl.grad(v)) * dx + \
     ufl.inner(beta, ufl.grad(u0)) * v * dx

A = (1/dt)*(ufl.inner(u, v) - ufl.inner(u0,v)) * dx + theta* A1 + (1-theta) * A0

r = (((1/dt)*(u-u0) + theta*((1.0/Pe)* ufl.div(ufl.grad(u)) + ufl.inner(beta,ufl.grad(u))) + 
      (1-theta)*((1.0/Pe)*ufl.div(ufl.grad(u0)) + ufl.inner(beta,ufl.grad(u0)))) * 
     tau * ufl.inner(beta,ufl.grad(v))*dx)

F = A + r - f * v * dx - f * tau * ufl.dot(beta, ufl.grad(v)) * dx

problem = LinearProblem(ufl.lhs(F), ufl.rhs(F), bcs=[bc_zero], 
                        petsc_options={"ksp_type": "gmres", "pc_type": "ilu"})
c = fem.Function(V)

xdmf = io.XDMFFile(domain.comm, "results_source_in_field.xdmf", "w")
xdmf.write_mesh(domain)

t = 0.0
c.x.array[:] = u0.x.array

while t < t_end:
    t += dt
    print(f"t = {t:.2f} von {t_end:.2f}")

    c = problem.solve()
    xdmf.write_function(c, t)
    u0.x.array[:] = c.x.array

xdmf.close()

x_coords = domain.geometry.x[:, 0]
y_coords = domain.geometry.x[:, 1]
c_values = c.x.array

plt.figure(figsize=(10, 8))
plt.scatter(x_coords, y_coords, c=c_values, cmap='viridis')
plt.colorbar(label='Konzentration (c)')
plt.title('Konzentration am Ende der Simulation')
plt.xlabel('X-Position')
plt.ylabel('Y-Position')
plt.show()