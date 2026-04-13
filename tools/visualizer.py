import os
import argparse
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.tri as mtri
from matplotlib.collections import LineCollection
from scipy.io import savemat
from scipy.spatial import cKDTree
from pathlib import Path
from basix.ufl import element
from dolfinx import fem, plot, mesh

class Visualizer:
    """
    2D visualizer for FEM meshes/fields using matplotlib.
    Useful for clean static plots of vector fields on domains (also with holes).
    """

    def __init__(self, function_space: fem.FunctionSpace, figsize=(10, 5), dpi=160):
        self.function_space = function_space
        self.mesh = function_space.mesh
        self.fig, self.ax = plt.subplots(figsize=figsize, dpi=dpi)
        self._last_mappable = None

        tdim = self.mesh.topology.dim
        self.mesh.topology.create_connectivity(tdim, 0)
        conn = self.mesh.topology.connectivity(tdim, 0).array

        num_cells = len(conn) // 3
        self.cells = conn.reshape(num_cells, 3)
        self.points = self.mesh.geometry.x[:, :2]

    def add_background_mesh(self, color="0.75", linewidth=0.25, alpha=0.9):
        x = self.points[:, 0]
        y = self.points[:, 1]
        self.ax.triplot(x, y, self.cells, color=color, linewidth=linewidth, alpha=alpha)

    def add_scalar_field(self, name: str, scalar_func: fem.Function, cmap: str = "coolwarm"):
        bs = scalar_func.function_space.dofmap.index_map_bs
        if not bs == 1:
            raise ValueError(f"{name} must be a scalar field (block size = 1).")

        V_plot = fem.functionspace(self.mesh, element("Lagrange", self.mesh.basix_cell(), 1))
        scalar_plot = fem.Function(V_plot)
        scalar_plot.interpolate(scalar_func)

        tri = mtri.Triangulation(self.points[:, 0], self.points[:, 1], self.cells)
        self._last_mappable = self.ax.tripcolor(
            tri,
            scalar_plot.x.array,
            shading="gouraud",
            cmap=cmap,
        )

    def add_vector_field(
        self,
        name: str,
        vector_func: fem.Function,
        stride: int = 2,
        cmap: str = "coolwarm",
        scale: float | None = None,
        width: float = 0.0022,
    ):
        bs = vector_func.function_space.dofmap.index_map_bs
        if bs < 2:
            raise ValueError(f"{name} must be a vector field with at least 2 components.")

        coords = vector_func.function_space.tabulate_dof_coordinates()[:, :2]
        values = vector_func.x.array.reshape(-1, bs)[:, :2]
        mag = np.linalg.norm(values, axis=1)

        stride = max(int(stride), 1)
        coords = coords[::stride]
        values = values[::stride]
        mag = mag[::stride]

        quiv = self.ax.quiver(
            coords[:, 0],
            coords[:, 1],
            values[:, 0],
            values[:, 1],
            mag,
            cmap=cmap,
            angles="xy",
            scale_units="xy",
            scale=scale,
            width=width,
            pivot="tail",
        )
        self._last_mappable = quiv

    def add_points(self, coords: np.ndarray, color="red", size=15, label: str | None = None):
        coords = np.asarray(coords)
        if coords.ndim == 1:
            coords = coords.reshape(1, -1)
        self.ax.scatter(coords[:, 0], coords[:, 1], c=color, s=size, label=label)

    def add_streamplot(
        self,
        name: str,
        vector_func: fem.Function,
        nx: int = 220,
        ny: int = 140,
        density: float = 1.6,
        cmap: str = "coolwarm",
        linewidth: float = 1.0,
        arrowsize: float = 1.0,
    ):
        """
        Plot streamlines for a 2D vector field on a regular grid.
        Grid points outside the mesh domain (including holes) are masked.
        """
        bs = vector_func.function_space.dofmap.index_map_bs
        if bs < 2:
            raise ValueError(f"{name} must be a vector field with at least 2 components.")

        # Regular plotting grid in domain bounding box
        x_min, y_min = np.min(self.points, axis=0)
        x_max, y_max = np.max(self.points, axis=0)
        xg = np.linspace(x_min, x_max, max(int(nx), 10))
        yg = np.linspace(y_min, y_max, max(int(ny), 10))
        xx, yy = np.meshgrid(xg, yg)
        q = np.column_stack([xx.ravel(), yy.ravel()])

        # Mark points outside mesh/hole regions
        tri = mtri.Triangulation(self.points[:, 0], self.points[:, 1], self.cells)
        tri_finder = tri.get_trifinder()
        inside = tri_finder(q[:, 0], q[:, 1]) >= 0

        # Interpolate from function DOF coordinates by nearest neighbor
        dof_xy = vector_func.function_space.tabulate_dof_coordinates()[:, :2]
        dof_uv = vector_func.x.array.reshape(-1, bs)[:, :2]
        tree = cKDTree(dof_xy)
        _, nn = tree.query(q[inside], k=1, workers=-1)

        U = np.full(q.shape[0], np.nan, dtype=float)
        V = np.full(q.shape[0], np.nan, dtype=float)
        U[inside] = dof_uv[nn, 0]
        V[inside] = dof_uv[nn, 1]

        U = U.reshape(xx.shape)
        V = V.reshape(xx.shape)
        speed = np.sqrt(U**2 + V**2)
        outside_mask = ~inside.reshape(xx.shape)

        strm = self.ax.streamplot(
            xg,
            yg,
            np.ma.array(U, mask=outside_mask),
            np.ma.array(V, mask=outside_mask),
            color=np.ma.array(speed, mask=outside_mask),
            cmap=cmap,
            density=density,
            linewidth=linewidth,
            arrowsize=arrowsize,
        )
        self._last_mappable = strm.lines

    def add_boundary_facets(
        self,
        facet_tags,
        tag_styles: dict[int, dict] | None = None,
        default_color: str = "black",
        default_linewidth: float = 1.6,
        default_linestyle: str = "-",
        alpha: float = 1.0,
    ):
        """Overlay tagged boundary facets as line segments on the active axes."""
        if facet_tags is None:
            return

        self.mesh.topology.create_connectivity(1, 0)
        f2v = self.mesh.topology.connectivity(1, 0)
        if f2v is None:
            raise RuntimeError("Mesh does not provide facet-to-vertex connectivity.")

        coords = self.mesh.geometry.x[:, :2]
        seen_labels = set()

        for tag in np.unique(facet_tags.values):
            facets = facet_tags.indices[facet_tags.values == tag]
            if len(facets) == 0:
                continue

            style = dict((tag_styles or {}).get(int(tag), {}))
            label = style.pop("label", None)
            color = style.pop("color", default_color)
            linewidth = style.pop("linewidth", default_linewidth)
            linestyle = style.pop("linestyle", default_linestyle)
            tag_alpha = style.pop("alpha", alpha)
            zorder = style.pop("zorder", 4)

            segments = []
            for facet in facets:
                vertices = f2v.links(int(facet))
                if len(vertices) != 2:
                    continue
                segments.append(coords[np.asarray(vertices, dtype=np.int32)])

            if not segments:
                continue

            collection = LineCollection(
                segments,
                colors=color,
                linewidths=linewidth,
                linestyles=linestyle,
                alpha=tag_alpha,
                zorder=zorder,
                **style,
            )
            if label is not None and label not in seen_labels:
                collection.set_label(label)
                seen_labels.add(label)
            self.ax.add_collection(collection)

    def show(
        self,
        title: str | None = None,
        filename: str | None = None,
        show_colorbar: bool = True,
        colorbar_label: str | None = None,
    ):
        self.ax.set_aspect("equal", adjustable="box")
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        if title is not None:
            self.ax.set_title(title)

        if show_colorbar and self._last_mappable is not None:
            cbar = self.fig.colorbar(self._last_mappable, ax=self.ax, pad=0.02)
            if colorbar_label is not None:
                cbar.set_label(colorbar_label)

        handles, labels = self.ax.get_legend_handles_labels()
        if labels:
            self.ax.legend(
                loc="upper center",
                bbox_to_anchor=(0.5, -0.14),
                ncol=max(1, len(labels)),
                frameon=False,
                borderaxespad=0.0,
            )

        self.fig.tight_layout()
        if filename is not None:
            self.fig.savefig(filename, dpi=self.fig.dpi)
            print(f"[Visualizer] Saved figure: {filename}")
        else:
            plt.show()

    
    ### csv plotting ###
    
    @staticmethod
    def plot_wind_slice_csv(
        csv_path: str | Path,
        z_height: float,
        z_tol: float = 0.05,
        output_path: str | Path | None = None,
        title: str | None = None,
        stride: int = 1,
        figsize: tuple[float, float] = (10, 5),
        dpi: int = 160,
        cmap: str = "coolwarm",
        scale: float | None = None,
        width: float = 0.0022,
        colorbar_label: str = "wind speed (m/s)",
        show: bool = True,
    ):
        csv_path = Path(csv_path).resolve()
        df = pd.read_csv(csv_path)
        gt_cols = ["Points:0", "Points:1", "Points:2", "U:0", "U:1"]

        if not all(col in df.columns for col in gt_cols):
            raise ValueError(
                f"Unsupported 3D wind CSV format in {csv_path.name}. "
                "Expected columns Points:0,Points:1,Points:2,U:0,U:1."
            )

        mask = np.abs(df["Points:2"].to_numpy(dtype=float) - float(z_height)) <= float(z_tol)
        df = df.loc[mask, gt_cols].copy()
        if df.empty:
            raise ValueError(
                f"No CSV rows found in slice z={z_height:.6g} +/- {z_tol:.6g} for {csv_path.name}."
            )

        df.columns = ["x", "y", "z", "wind_x", "wind_y"]
        stride = max(int(stride), 1)
        df = df.iloc[::stride].copy()

        x = df["x"].to_numpy(dtype=float)
        y = df["y"].to_numpy(dtype=float)
        u = df["wind_x"].to_numpy(dtype=float)
        v = df["wind_y"].to_numpy(dtype=float)
        speed = np.sqrt(u * u + v * v)

        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        quiv = ax.quiver(x, y, u, v, speed,
            cmap=cmap,
            angles="xy",
            scale_units="xy",
            scale=scale,
            width=width,
            pivot="tail",
        )
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")

        if title is not None:
            ax.set_title(title)

        cbar = fig.colorbar(quiv, ax=ax, pad=0.02)
        if colorbar_label is not None:
            cbar.set_label(colorbar_label)

        fig.tight_layout()

        if output_path is not None:
            output_path = Path(output_path)
            fig.savefig(output_path, dpi=dpi)
            print(f"[plot_wind_csv_slice] Saved figure: {output_path}")

        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig, ax

    @staticmethod
    def plot_wind_2Dcsv(
        csv_path: str | Path,
        output_path: str | Path | None = None,
        title: str | None = None,
        stride: int = 1,
        figsize: tuple[float, float] = (10, 5),
        dpi: int = 160,
        cmap: str = "coolwarm",
        scale: float | None = None,
        width: float = 0.0022,
        colorbar_label: str = "speed",
        show: bool = True,
    ):
        csv_path = Path(csv_path).resolve()
        df = pd.read_csv(csv_path)

        simple_cols = ["x", "y", "wind_x", "wind_y"]
        gt_cols = ["Points:0", "Points:1", "U:0", "U:1"]

        if all(col in df.columns for col in simple_cols):
            df = df[simple_cols].copy()
        elif all(col in df.columns for col in gt_cols):
            df = df[gt_cols].copy()
            df.columns = simple_cols
        else:
            raise ValueError(
                f"Unsupported wind CSV format in {csv_path.name}. "
                "Expected either columns x,y,wind_x,wind_y or Points:0,Points:1,U:0,U:1."
            )
        stride = max(int(stride), 1)
        df = df.iloc[::stride].copy()

        x = df["x"].to_numpy(dtype=float)
        y = df["y"].to_numpy(dtype=float)
        u = df["wind_x"].to_numpy(dtype=float)
        v = df["wind_y"].to_numpy(dtype=float)
        speed = np.sqrt(u * u + v * v)

        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        quiv = ax.quiver(x, y, u, v, speed,
            cmap=cmap,
            angles="xy",
            scale_units="xy",
            scale=scale,
            width=width,
            pivot="tail",
        )
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")

        if title is not None:
            ax.set_title(title)

        cbar = fig.colorbar(quiv, ax=ax, pad=0.02)
        if colorbar_label is not None:
            cbar.set_label(colorbar_label)

        fig.tight_layout()

        if output_path is not None:
            output_path = Path(output_path)
            fig.savefig(output_path, dpi=dpi)
            print(f"[plot_wind_csv] Saved figure: {output_path}")

        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig, ax

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize wind CSV files and save them as PNG plots.")
    parser.add_argument("windfile_csv", type=str, help="Path to the wind CSV file.")
    parser.add_argument("-o", "--output-dir", type=str, default=None, help="Optional output directory for the PNG. Defaults to the input file's parent directory.")
    parser.add_argument("--title", type=str, default=None, help="Optional plot title.")
    parser.add_argument("--stride", type=int, default=1, help="Plot every n-th row to reduce clutter.")
    parser.add_argument("--scale", type=float, default=None, help="Optional matplotlib quiver scale.")
    parser.add_argument("--width", type=float, default=0.0022, help="Arrow width for quiver plots.")
    parser.add_argument("--dpi", type=int, default=160, help="Figure DPI.")
    parser.add_argument("--figsize", type=float, nargs=2, metavar=("W", "H"), default=(10, 5), help="Figure size in inches.")
    parser.add_argument("--cmap", type=str, default="coolwarm", help="Matplotlib colormap.")
    parser.add_argument("--show", action="store_true", help="Also show the figure interactively.")
    parser.add_argument("--z-height", type=float, default=None, help="Slice center height for 3D CSV files.")
    parser.add_argument("--z-tol", type=float, default=0.05, help="Half-thickness of the z slice, e.g. 0.05 means +/- 5 cm.")
    parser.add_argument("--z-span-threshold", type=float, default=0.2, help="If the z-span exceeds this threshold, the CSV is treated as 3D.")
    args = parser.parse_args()

    csv_path = Path(args.windfile_csv).resolve()
    if csv_path.suffix.lower() != ".csv":
        raise ValueError("Can only visualize .csv files.")

    output_dir = csv_path.parent if args.output_dir is None else Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    df_head = pd.read_csv(csv_path, nrows=1)
    if "Points:2" in df_head.columns:
        z = pd.read_csv(csv_path, usecols=["Points:2"])["Points:2"].to_numpy(dtype=float)
        is_3d = z.size > 0 and float(np.max(z) - np.min(z)) > float(args.z_span_threshold)
    else:
        is_3d = False
    if is_3d:
        if args.z_height is None:
            raise ValueError(
                f"{csv_path.name} is treated as a 3D CSV because its z-span exceeds {args.z_span_threshold:.3g} m. "
                "Pass --z-height to choose the slice to plot."
            )
        output_path = output_dir / f"{csv_path.stem}_z{args.z_height:g}.png"
        Visualizer.plot_wind_slice_csv(
            csv_path,
            z_height=args.z_height,
            z_tol=args.z_tol,
            output_path=output_path,
            title=args.title,
            stride=args.stride,
            figsize=tuple(args.figsize),
            dpi=args.dpi,
            cmap=args.cmap,
            scale=args.scale,
            width=args.width,
            show=args.show,
        )
    else:
        output_path = output_dir / f"{csv_path.stem}.png"
        Visualizer.plot_wind_2Dcsv(
            csv_path,
            output_path=output_path,
            title=args.title,
            stride=args.stride,
            figsize=tuple(args.figsize),
            dpi=args.dpi,
            cmap=args.cmap,
            scale=args.scale,
            width=args.width,
            show=args.show,
        )

