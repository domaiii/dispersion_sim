import gmsh
from pathlib import Path
from dolfinx import plot
import dolfinx.io as dio
from mpi4py import MPI
import pyvista as pv

meshfile = Path("/app/meshes/mesh_mixed_obstacles/mesh.msh").resolve()

domain, cell_tags, facet_tags = dio.gmshio.read_from_msh(meshfile, MPI.COMM_WORLD, gdim=2)

topology, cell_type, geom = plot.vtk_mesh(domain) 
grid = pv.UnstructuredGrid(topology, cell_type, geom)

grid.plot(screenshot="mesh_vis.png", off_screen=True, show_edges=True, cpos="xy")
