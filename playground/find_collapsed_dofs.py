import numpy as np
from mpi4py import MPI
from dolfinx import fem, mesh, plot
from basix.ufl import element, mixed_element
import pyvista as pv


# --- Helper to visualize velocity BCs ---
def visualize_bc(w, scale=0.1):
    u = w.sub(0).collapse()
    grid = pv.UnstructuredGrid(*plot.vtk_mesh(u.function_space))
    vals = np.zeros((len(u.x.array)//2, 3))
    vals[:, :2] = u.x.array.reshape(-1, 2)
    grid["u"] = vals
    pl = pv.Plotter()
    pl.add_mesh(grid, color="lightgray", opacity=0.4)
    pl.add_mesh(grid.glyph(orient="u", factor=scale))
    pl.view_xy()
    pl.show()


# --- Mesh and spaces ---
domain = mesh.create_unit_square(MPI.COMM_WORLD, 10, 5)
elem_u = element("Lagrange", domain.basix_cell(), 2, shape=(2,))
elem_p = element("Lagrange", domain.basix_cell(), 1)
W = fem.functionspace(domain, mixed_element([elem_u, elem_p]))
W0 = W.sub(0)
V, V_to_W = W0.collapse()

# --- Inflow boundary (x = 0) ---
facets = mesh.locate_entities_boundary(domain, 1, lambda x: np.isclose(x[0], 0))
inflow_dofs_W0 = fem.locate_dofs_topological((W0, V), 1, facets)

# --- Works: BC defined via (W0, V) ---
u2 = fem.Function(V)
u2.interpolate(lambda x: (np.ones_like(x[0]), np.zeros_like(x[0]))) # set only x component
bc2 = fem.dirichletbc(u2, inflow_dofs_W0, W0)

# tmp2 = fem.Function(W)
# bc2.set(tmp2.x.array)
# tmp2.x.scatter_forward()
# visualize_bc(tmp2)  # no inflow visible

u2.x.array[inflow_dofs_W0[1]] += 1.0 # modify x and y component afterwards!

# tmp2 = fem.Function(W) 
# bc2.set(tmp2.x.array)
# tmp2.x.scatter_forward()
# visualize_bc(tmp2)


def visualize_dofs(V: fem.FunctionSpace, domain: mesh.Mesh, dof_subset):
    coords = V.tabulate_dof_coordinates()
    # DOF-Koordinaten der Auswahl holen und Duplikate (x/y der gleichen Stelle) zusammenfassen
    selected_coords = np.unique(coords[dof_subset], axis=0)

    topology, cell_type, geom = plot.vtk_mesh(domain)
    grid = pv.UnstructuredGrid(topology, cell_type, geom)
    dof_points = pv.PolyData(selected_coords)

    pl = pv.Plotter()
    pl.add_mesh(grid, color="lightgray", opacity=0.3, show_edges=True)
    pl.add_mesh(dof_points, color="red", point_size=10, render_points_as_spheres=True)
    pl.view_xy()
    pl.show()

visualize_dofs(V, domain, inflow_dofs_W0[1])