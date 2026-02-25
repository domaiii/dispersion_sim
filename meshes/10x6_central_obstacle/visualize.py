import gmsh
import numpy as np
from basix.ufl import element
from pathlib import Path
from dolfinx import fem
from dolfinx import plot
from dolfinx.mesh import MeshTags
import dolfinx.io as dio
from mpi4py import MPI
import pyvista as pv

meshfile = Path("/app/meshes/10x6_central_obstacle/mesh.msh").resolve()

domain, cell_tags, facet_tags = dio.gmshio.read_from_msh(meshfile, MPI.COMM_WORLD, gdim=2)

topology, cell_type, geom = plot.vtk_mesh(domain) 
grid = pv.UnstructuredGrid(topology, cell_type, geom)

grid.plot(screenshot="mesh_vis.png", off_screen=True, show_edges=True, cpos="xy")

elem = element("Lagrange", domain.basix_cell(), 2, shape=(2,))

space = fem.functionspace(domain, elem)
fun = fem.Function(space)