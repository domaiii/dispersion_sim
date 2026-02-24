import gmsh
from pathlib import Path
from dolfinx import plot
import dolfinx.io as dio
from mpi4py import MPI
import pyvista as pv

meshfile = Path("/home/dominik/git/dispersion_sim/meshes/mesh_mixed_obstacles/mesh.msh").resolve()

domain, cell_tags, facet_tags = dio.gmshio.read_from_msh(meshfile, MPI.COMM_WORLD, gdim=2)

topology, cell_type, geom = plot.vtk_mesh(domain) 
grid = pv.UnstructuredGrid(topology, cell_type, geom)

plotter = pv.Plotter()
plotter.add_mesh(grid)
plotter.view_xy()
plotter.show()

