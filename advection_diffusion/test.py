import numpy as np
import pyvista as pv
from mpi4py import MPI
from dolfinx import mesh, plot

nx = 128
ny = 128

domain = mesh.create_unit_square(MPI.COMM_WORLD, nx, ny)

topology, cell_type, geom = plot.vtk_mesh(domain) 
grid = pv.UnstructuredGrid(topology, cell_type, geom)

f = np.random.rand(nx+1, ny+1)
grid.point_data["f"] = f.reshape(-1, 1)
plotter = pv.Plotter()
plotter.add_mesh(
    grid, 
    scalar_bar_args={
        "vertical": True,
        "position_x": 0.85,
        "position_y": 0.2,
        "height": 0.6,
        "width": 0.08   }
)
plotter.show_grid(xtitle="X", ytitle="Y")
plotter.view_xy()
plotter.show()