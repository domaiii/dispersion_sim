from dolfinx import fem
from pathlib import Path
import scipy
import dolfinx.io as dio
import pyvista as pv
import pandas as pd
import numpy as np
from basix.ufl import element
from mpi4py import MPI
from dolfinx import plot

from visualizer import MatplotlibVisualizer2D

def load_from_3Dcsv(
    wind_csv: str | Path,
    height: float,
    z_tol: float,
    dest: fem.Function,
    max_xy_dist: float | None = None,
):
    """
    Load 2D wind layer from 3D csv into given FEM Function.
    """
    df = pd.read_csv(wind_csv)
    z = df["Points:2"].to_numpy()
    z_min = float(np.min(z))
    z_max = float(np.max(z))
    if not (z_min <= height <= z_max):
        raise ValueError(
            f"height={height} is outside CSV z-range [{z_min}, {z_max}]"
        )

    layer = df[np.abs(z - height) <= z_tol]
    if layer.empty:
        raise ValueError(f"No points in z-slice: height={height}, z_tol={z_tol}")

    src_xy = layer[["Points:0", "Points:1"]].to_numpy()
    src_u = layer[["U:0", "U:1"]].to_numpy()

    nodes_xy = dest.function_space.tabulate_dof_coordinates()[:, :2]
    x_min = float(np.min(df["Points:0"].to_numpy()))
    x_max = float(np.max(df["Points:0"].to_numpy()))
    y_min = float(np.min(df["Points:1"].to_numpy()))
    y_max = float(np.max(df["Points:1"].to_numpy()))
    tol_x = 0.01 * x_max
    tol_y = 0.01 * y_max
    
    outside = (
        (nodes_xy[:, 0] < x_min - tol_x) | (nodes_xy[:, 0] > x_max + tol_x) |
        (nodes_xy[:, 1] < y_min - tol_y) | (nodes_xy[:, 1] > y_max + tol_y)
    )
    if np.any(outside):
        n_out = int(np.count_nonzero(outside))
        raise ValueError(
            f"{n_out} target nodes are outside CSV xy-range "
            f"x:[{x_min}, {x_max}], y:[{y_min}, {y_max}]"
        )

    tree = scipy.spatial.cKDTree(src_xy)
    dist, idx = tree.query(nodes_xy, k=1, p=2.0, workers=-1)

    max_dist = float(np.max(dist))

    if max_xy_dist is not None and max_dist > max_xy_dist:
        raise ValueError(
            f"Maximum XY interpolation distance exceeded: "
            f"{max_dist:.6g} > {max_xy_dist:.6g}"
        )
    
    wind_vals = src_u[idx]

    bs = dest.function_space.dofmap.index_map_bs
    if bs < 2:
        raise ValueError("dest must be vector-valued with at least 2 components")

    arr = dest.x.array.reshape(-1, bs)
    arr[:, 0] = wind_vals[:, 0]
    arr[:, 1] = wind_vals[:, 1]
    dest.x.scatter_forward()

if __name__=="__main__":

    meshfile = Path("/app/meshes/10x6_central_obstacle/mesh.msh").resolve()
    domain, cell_tags, facet_tags = dio.gmshio.read_from_msh(meshfile, MPI.COMM_WORLD, gdim=2)
    elem = element("Lagrange", domain.basix_cell(), 2, shape=(2,))
    space = fem.functionspace(domain, elem)

    fun = fem.Function(space)

    height = 1.0 # m
    tol = 0.1 # m
    csv_file = "/app/meshes/10x6_central_obstacle/wind_solution.csv"

    load_from_3Dcsv(
        csv_file,
        height,
        tol,
        fun,
        debug_plot_file="/app/tools/outside_nodes_debug.png",
        dist_debug_plot_file="/app/tools/large_distance_nodes_debug.png",
        dist_threshold=0.2,
    )

    vis = MatplotlibVisualizer2D(space)
    vis.add_background_mesh()
    #vis.add_vector_field("wind", fun, stride=1, scale=1.5)
    vis.add_streamplot("stream", fun, 200, 100, 1.5)
    vis.show("matplotlib plot", "matplotlib_wind.png")