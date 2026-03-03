import scipy
import adios4dolfinx
import dolfinx.io as dio
import pyvista as pv
import pandas as pd
import numpy as np

from dolfinx import fem
from pathlib import Path
from basix.ufl import element
from mpi4py import MPI
from dolfinx import plot

from visualizer import MatplotlibVisualizer2D


def create_sample_points_with_wind_csv(
    wind_csv: str | Path,
    n_points: int,
    seed: int,
    z_height: float,
    z_tol: float,
    output_dir: str | Path | None = None,
) -> Path:
    """
    Sample random points from 3D wind ground-truth CSV and save output CSV:
    sample_id, x, y, z, wind_x, wind_y, wind_z.
    """
    if n_points <= 0:
        raise ValueError("n_points must be > 0.")

    wind_csv = Path(wind_csv).resolve()
    if not wind_csv.exists():
        raise FileNotFoundError(f"Wind CSV not found: {wind_csv}")

    df = pd.read_csv(wind_csv)
    required_cols = ["Points:0", "Points:1", "Points:2", "U:0", "U:1", "U:2"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns in {wind_csv.name}: {missing}."
        )

    rows = df[required_cols].copy()
    rows = rows[np.abs(rows["Points:2"].to_numpy() - z_height) <= z_tol]

    if len(rows) < n_points:
        raise ValueError(
            f"Requested n_points={n_points}, but only {len(rows)} valid rows available."
        )

    rng = np.random.default_rng(seed)
    sample_idx = rng.choice(len(rows), size=n_points, replace=False)
    sampled = rows.iloc[sample_idx].reset_index(drop=True)

    out = pd.DataFrame(
        {
            "sample_id": np.arange(n_points, dtype=np.int32),
            "x": sampled["Points:0"].to_numpy(),
            "y": sampled["Points:1"].to_numpy(),
            "z": sampled["Points:2"].to_numpy(),
            "wind_x": sampled["U:0"].to_numpy(),
            "wind_y": sampled["U:1"].to_numpy(),
            "wind_z": sampled["U:2"].to_numpy(),
        }
    )

    if output_dir is None:
        output_dir = wind_csv.parent
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    out_path = output_dir / f"sample_points_n{n_points}_seed{seed}.csv"
    out.to_csv(out_path, index=False)
    return out_path


def read_sample_points_with_wind_csv(samples_csv: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Read sample CSV format:
    sample_id,x,y,z,wind_x,wind_y,wind_z

    Returns
    -------
    xy : np.ndarray
        shape (N, 2) with sample coordinates.
    uv : np.ndarray
        shape (N, 2) with observed wind components.
    """
    samples_csv = Path(samples_csv).resolve()
    if not samples_csv.exists():
        raise FileNotFoundError(f"Samples CSV not found: {samples_csv}")

    df = pd.read_csv(samples_csv)
    required = ["x", "y", "wind_x", "wind_y"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns in {samples_csv.name}: {missing}. "
            f"Expected at least {required}."
        )

    xy = df[["x", "y"]].to_numpy(dtype=float)
    uv = df[["wind_x", "wind_y"]].to_numpy(dtype=float)

    if len(xy) == 0:
        raise ValueError(f"No sample rows found in {samples_csv}")

    return xy, uv


def map_xy_to_nearest(
    src_xy: np.ndarray,
    query_xy: np.ndarray,
    max_xy_dist: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Map each query point to nearest source point in XY.

    Returns
    -------
    dist : np.ndarray
        shape (M,), nearest-neighbor distances.
    idx : np.ndarray
        shape (M,), index into src_xy for each query point.
    """
    src_xy = np.asarray(src_xy, dtype=float)
    query_xy = np.asarray(query_xy, dtype=float)

    if src_xy.ndim != 2 or src_xy.shape[1] != 2:
        raise ValueError("src_xy must have shape (N, 2).")
    if query_xy.ndim != 2 or query_xy.shape[1] != 2:
        raise ValueError("query_xy must have shape (M, 2).")
    if len(src_xy) == 0:
        raise ValueError("src_xy is empty.")
    if len(query_xy) == 0:
        raise ValueError("query_xy is empty.")

    tree = scipy.spatial.cKDTree(src_xy)
    dist, idx = tree.query(query_xy, k=1, p=2.0, workers=-1)

    max_dist = float(np.max(dist))
    if max_xy_dist is not None and max_dist > max_xy_dist:
        raise ValueError(
            f"Maximum XY mapping distance exceeded: {max_dist:.6g} > {max_xy_dist:.6g}"
        )

    return dist, idx


def map_samples_csv_to_measurements(
    samples_csv: str | Path,
    velocity_node_xy: np.ndarray,
    V_to_W: np.ndarray,
    unique_nodes: bool = True,
    max_xy_dist: float | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """
    Convert sample CSV to measurement arrays for AirflowEstimator.

    Returns
    -------
    measurement_ids_W : np.ndarray
        Flattened W-indices (ux,uy dofs).
    measurement_values : np.ndarray
        Flattened values aligned with measurement_ids_W.
    info : dict[str, float]
        Basic mapping stats.
    """
    samples_xy, samples_uv = read_sample_points_with_wind_csv(samples_csv)

    dist, node_ids = map_xy_to_nearest(
        src_xy=np.asarray(velocity_node_xy, dtype=float)[:, :2],
        query_xy=samples_xy,
        max_xy_dist=max_xy_dist,
    )

    n_input = int(len(node_ids))
    n_dropped = 0
    if unique_nodes:
        _, first_idx = np.unique(node_ids, return_index=True)
        keep = np.sort(first_idx)
        n_dropped = n_input - int(len(keep))
        node_ids = node_ids[keep]
        samples_uv = samples_uv[keep]
        dist = dist[keep]

    x_ids = node_ids * 2
    y_ids = node_ids * 2 + 1
    velocity_ids_V = np.stack((x_ids, y_ids)).T.flatten().astype(np.int32)
    measurement_ids_W = np.asarray(V_to_W, dtype=np.int32)[velocity_ids_V]
    measurement_values = np.stack((samples_uv[:, 0], samples_uv[:, 1]), axis=1).flatten()

    info = {
        "n_input_samples": float(n_input),
        "n_used_samples": float(len(node_ids)),
        "n_dropped_duplicate_nodes": float(n_dropped),
        "max_xy_dist": float(np.max(dist)) if len(dist) else 0.0,
    }
    return measurement_ids_W, measurement_values, info


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

    _, idx = map_xy_to_nearest(
        src_xy=src_xy,
        query_xy=nodes_xy,
        max_xy_dist=max_xy_dist,
    )

    wind_vals = src_u[idx]

    bs = dest.function_space.dofmap.index_map_bs
    if bs < 2:
        raise ValueError("dest must be vector-valued with at least 2 components")

    arr = dest.x.array.reshape(-1, bs)
    arr[:, 0] = wind_vals[:, 0]
    arr[:, 1] = wind_vals[:, 1]
    dest.x.scatter_forward()


if __name__ == "__main__":

    meshfile = Path("/app/meshes/10x6_central_obstacle/mesh.msh").resolve()
    domain, cell_tags, facet_tags = dio.gmshio.read_from_msh(meshfile, MPI.COMM_WORLD, gdim=2)
    elem = element("Lagrange", domain.basix_cell(), 2, shape=(2,))
    space = fem.functionspace(domain, elem)

    fun = fem.Function(space)

    height = 1.0  # m
    tol = 0.1  # m
    csv_file = "/app/meshes/10x6_central_obstacle/wind_solution.csv"

    load_from_3Dcsv(
        csv_file,
        height,
        tol,
        fun,
        max_xy_dist=0.2
    )

    vis = MatplotlibVisualizer2D(space)
    vis.add_background_mesh()
    vis.add_streamplot("stream", fun, 200, 100, 1.5)
    vis.show("matplotlib plot", "matplotlib_wind.png")

    wind_file = Path("/app/exp_sample_based_estimation/exp_wind_comparison/airflow_10x6_ground_truth.bp")
    adios4dolfinx.write_mesh(wind_file, domain)
    adios4dolfinx.write_meshtags(wind_file, domain, facet_tags, meshtag_name="facet_tags")
    adios4dolfinx.write_function(wind_file, fun, name="velocity_H2")
