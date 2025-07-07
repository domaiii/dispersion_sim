from dolfinx import fem, mesh
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
from mpi4py import MPI
import numpy as np
import ufl
from basix.ufl import element, mixed_element
from petsc4py.PETSc import ScalarType, KSP
from petsc4py import PETSc

x_lim = 2.0
y_lim = 1.0
pts = ((0.0, 0.0), (x_lim, y_lim))
domain = mesh.create_rectangle(MPI.COMM_WORLD, pts, (32, 16), mesh.CellType.quadrilateral)

elem_u = element("Lagrange", domain.basix_cell(), 2, shape=(domain.topology.dim,))
elem_p = element("Lagrange", domain.basix_cell(), 1)

mixed_e = mixed_element([elem_u, elem_p])

V = fem.functionspace(domain, mixed_e)

u, p = ufl.TrialFunctions(V)
v, q = ufl.TestFunctions(V)

w = fem.Function(V)
w.x.array[:] = 0.0
u_n, p_n = ufl.split(w)
nu = fem.Constant(domain, ScalarType(14.9e-6)) # m²/s

# nochmal nachvollziehen wie NonLinearProblem zustande kommt
F = (ufl.inner(ufl.dot(u_n, ufl.nabla_grad(u_n)), v) * ufl.dx
     + ufl.inner(nu * ufl.grad(u_n), ufl.grad(v)) * ufl.dx
     - ufl.inner(p_n, ufl.div(v)) * ufl.dx
     - ufl.inner(ufl.div(u_n), q) * ufl.dx)

J = ufl.derivative(F, w, ufl.TrialFunction(V))

V0, submap = V.sub(0).collapse()
V1, submap1 = V.sub(1).collapse()

# boundaries
def walls(x):
    return np.logical_or(np.isclose(x[1], 0), np.isclose(x[1], y_lim))

wall_dofs = fem.locate_dofs_geometrical((V.sub(0), V0), walls)
zero_velocity = fem.Function(V0)
zero_velocity.x.array[:] = 0.0
bc_noslip_walls = fem.dirichletbc(zero_velocity, wall_dofs, V.sub(0))

def inflow(x):
    return np.isclose(x[0], 0)

def inflow_velocity(x):
    ux_val = 4.0 * (y_lim - x[1]) * x[1] / y_lim**2
    return np.stack((ux_val, np.zeros_like(x[1])))

inflow_int = fem.Function(V0)
inflow_int.interpolate(inflow_velocity)
inlet_dofs = fem.locate_dofs_geometrical((V.sub(0), V0), inflow)
bc_inlet = fem.dirichletbc(inflow_int, inlet_dofs, V.sub(0))

def outflow(x):
    return np.isclose(x[0], x_lim)

zero_pressure_func = fem.Function(V1)
zero_pressure_func.x.array[:] = 0.0
outlet_dofs = fem.locate_dofs_geometrical((V.sub(1), V1), outflow)
bc_outlet = fem.dirichletbc(zero_pressure_func, outlet_dofs, V.sub(1))

bcs = [bc_noslip_walls, bc_inlet, bc_outlet]

# Erstelle ein NonlinearProblem Objekt
problem = NonlinearProblem(F, w, bcs, J, form_compiler_options={"quadrature_rule": "default"})

# Erstelle den NewtonSolver
solver = NewtonSolver(MPI.COMM_WORLD, problem)

# Konfiguration des internen KSP-Solvers direkt
breakpoint()
ksp = solver.krylov_solver
ksp.setType(PETSc.KSP.Type.GMRES)
ksp.getPC().setType(PETSc.PC.Type.LU)

ksp.setTolerances(rtol=1e-9, atol=1e-10)
ksp.setFromOptions() # Hier ist es sinnvoll, falls über Kommandozeile weitere Optionen kommen


# Setze Newton-Solver Toleranzen (für die nichtlineare Iteration)
solver.atol = 1e-8
solver.rtol = 1e-8
solver.max_it = 50

# Löse das System
(num_its, converged) = solver.solve(w)

print(f"Number of Newton iterations: {num_its}")
print(f"Converged: {converged}")


V_u_vis = fem.functionspace(domain, element("Lagrange", domain.basix_cell(), 1, shape=(domain.topology.dim,)))
V_p_vis = fem.functionspace(domain, element("Lagrange", domain.basix_cell(), 1))

# Erstelle neue fem.Function-Objekte, in die wir interpolieren
u_vis = fem.Function(V_u_vis)
p_vis = fem.Function(V_p_vis)

# --- KRITISCHE KORREKTUR HIER: direkter Zugriff auf w.sub(i) und Interpolation ---
# Splitten Sie 'w' in seine Komponenten als Function-Objekte
w_u, w_p = w.split() # This gives dolfinx.fem.Function objects for sub-spaces

# Interpoliere die Velocity-Komponente von w in u_vis
u_vis.interpolate(w_u) # Now passing a dolfinx.fem.Function directly

# Interpoliere die Pressure-Komponente von w in p_vis
p_vis.interpolate(w_p) # Now passing a dolfinx.fem.Function directly


# 2. Ergebnisse im XDMF-Format speichern
from dolfinx.io import XDMFFile
from pathlib import Path

# Sicherstellen, dass ein Ausgabeordner existiert
output_dir = Path("output_results")
output_dir.mkdir(exist_ok=True)

# Dateinamen für Geschwindigkeit und Druck
velocity_file = output_dir / "velocity.xdmf"
pressure_file = output_dir / "pressure.xdmf"

# Geschwindigkeit speichern
with XDMFFile(MPI.COMM_WORLD, velocity_file, "w") as f_u:
    f_u.write_mesh(domain)
    f_u.write_function(u_vis)

# Druck speichern
with XDMFFile(MPI.COMM_WORLD, pressure_file, "w") as f_p:
    f_p.write_mesh(domain)
    f_p.write_function(p_vis)

print(f"Ergebnisse in '{output_dir}' gespeichert: velocity.xdmf und pressure.xdmf")