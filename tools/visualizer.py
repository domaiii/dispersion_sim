import os
import numpy as np
import pyvista as pv
from scipy.io import savemat
from pathlib import Path
from dolfinx import fem, plot, mesh

class Visualizer2D:

    def __init__(self, function_space: fem.FunctionSpace, window_size=(1600, 900), font_size=16):
        self.function_space = function_space
        self.topology, self.cell_type, self.geom = plot.vtk_mesh(function_space)
        self.grid = pv.UnstructuredGrid(self.topology, self.cell_type, self.geom)
        self.has_points = False

        head = os.environ.get("DISPLAY") is None or os.environ.get("PYVISTA_OFF_SCREEN") == "true"
        self.plotter = pv.Plotter(window_size=window_size, off_screen=head)
        
        # Tracks the active scalar bar actor for removal (ensures only one is shown)
        self._active_scalar_bar_actor = None
        self._active_scalar_mesh = None 
        
        self._configure_style(font_size)

    def _configure_style(self, font_size):
        """Sets unified fonts and colorbar style."""
        self.scalar_bar_args = dict(
            title_font_size=font_size + 2,
            label_font_size=font_size,
            n_labels=5,
            position_x=0.3,
            position_y=0.05,
            width=0.4,
            height=0.03,
            fmt="%.2f"
        )
        
    def add_scalar_field(self, name: str, scalar_func: fem.Function, cmap: str = "coolwarm"):
        """
        Adds a scalar field (heatmap) and manages the colorbar. 
        Only one scalar field is visible at any time.
        """
        value_size = scalar_func.x.block_size
        if value_size != 1:
            raise ValueError(f"{name} is no scalar field (value_size={value_size}) — ignored.")

        # Overwrite previous if necessary
        if self._active_scalar_bar_actor is not None:
            self.plotter.remove_actor(self._active_scalar_bar_actor)
            if self._active_scalar_mesh is not None:
                self.plotter.remove_actor(self._active_scalar_mesh)

        # Add scalar data to the underlying grid
        self.grid.point_data[name] = scalar_func.x.array
        
        # Add a new mesh with the new scalar data
        actor = self.plotter.add_mesh(
            self.grid.copy(),
            scalars=name,
            cmap=cmap,
            scalar_bar_args={**self.scalar_bar_args, "title": name},
            render=False 
        )
        
        self._active_scalar_bar_actor = self.plotter.scalar_bar
        self._active_scalar_mesh = actor 


    def add_vector_field(self, name: str, vector_func: fem.Function, factor: float | None = None):
        """
        Adds a vector field visualization using glyphs (arrows).
        """
        if factor is None:
            factor = 1.0

        vec2d = vector_func.x.array.reshape(-1, 2)
        vec3d = np.hstack((vec2d, np.zeros((vec2d.shape[0], 1))))
        self.grid.point_data[name] = vec3d

        subset = self.grid.extract_points(np.arange(0, self.grid.n_points, 2), include_cells=False)
        glyphs = subset.glyph(orient=name, scale=name, factor=factor)
        
        self.plotter.add_mesh(glyphs, cmap="coolwarm", 
                              scalar_bar_args={**self.scalar_bar_args, "title": f"{name} Magnitude"})

    def add_points(self, coords: list | tuple | np.ndarray, 
                   color: str | None = None, size: int | None = None, label: str | None = None):
        """
        Add 2D or 3D point coordinates. Accepts list, tuple, or ndarray.
        """
        # Convert list/tuple -> ndarray
        coords = np.asarray(coords)

        # Guarantee correct shape
        if coords.ndim == 1:
            coords = coords.reshape(1, -1)

        # Convert to 3D if needed
        if coords.shape[1] == 2:
            coords = np.column_stack([coords, np.zeros(len(coords))])

        if size is None: size = 10
        if color is None: color = "red"
        if label is not None: self.has_points = True
        
        pts = pv.PolyData(coords)
        self.plotter.add_mesh(
            pts, 
            color=color, 
            point_size=size, 
            label=label
        )

    def add_background_mesh(self, opacity: float = 0.3, gridlines: bool = False):
        """
        Adds the underlying computational mesh as a background.
        """
        self.plotter.add_mesh(self.grid, color="gray", opacity=opacity, show_edges=gridlines)

    def show(self, title: str = None, zoom: float = 1.0, filename: str = "plot_output.png"):
            """
            Show the plot or save it to file if no graphical interface is available.
            """
            self.plotter.view_xy()
            self.plotter.add_axes()
            
            if self.has_points:
                self.plotter.add_legend(face="circle", size=(0.15, 0.1)) 
                
            if title:
                self.plotter.add_text(title, position="upper_edge", font_size=16, color="black")
            
            self.plotter.zoom_camera(zoom)

            if self.plotter.off_screen:
                out_path = Path(filename)
                if out_path.exists():
                    stem = out_path.stem
                    suffix = out_path.suffix
                    parent = out_path.parent
                    candidate = parent / f"{stem}_1{suffix}"
                    idx = 2
                    while candidate.exists():
                        candidate = parent / f"{stem}_{idx}{suffix}"
                        idx += 1
                    out_path = candidate

                print(f"[Visualizer2D] Headless Mode: Saving screenshot to {out_path}")
                os.environ['GALLIUM_DRIVER'] = 'llvmpipe'
                pv.start_xvfb()
                self.plotter.show(screenshot=str(out_path))
            else:
                self.plotter.show()

                  

    @staticmethod
    def export_function_matlab(f: fem.Function, filename: str | Path):
        """
        Export a dolfinx Function (scalar or vector) and the P1 mesh to a MATLAB .mat file.

        Supports:
            - scalar fields:  f : Ω -> R
            - vector fields:  f : Ω -> R^2 (first 2 comps)
            on triangular meshes.
        """
        mesh = f.function_space.mesh
        tdim = mesh.topology.dim

        # --- Ensure P1 triangle connectivity exists ---
        mesh.topology.create_connectivity(tdim, 0)
        conn = mesh.topology.connectivity(tdim, 0).array
        num_cells = len(conn) // 3
        cells = conn.reshape(num_cells, 3) + 1    # MATLAB = 1-based

        # --- Extract geometry (2D only) ---
        points = mesh.geometry.x[:, :2]

        # --- Extract function values ---
        arr = f.x.array
        block = f.x.block_size

        if block == 1:
            # scalar function
            U = arr.reshape(-1, 1)
        elif block >= 2:
            # vector function → take first two components
            U = arr.reshape(-1, block)[:, :2]
        else:
            raise ValueError(f"Unsupported block size: {block}")

        data = {
            "cells": cells.astype(np.int32),
            "points": points.astype(float),
            "u": U.astype(float)
        }

        savemat(filename, data)
        print(f"[export_matlab] Wrote MATLAB file: {filename}")

    @staticmethod
    def export_domain_matlab(mesh: mesh.Mesh, filename: str | Path, facet_tags=None):
        """
        Export a dolfinx 2D triangular P1 mesh to MATLAB.
        
        Saves:
            - cells : (Nc x 3) int32   triangle connectivity (1-based)
            - points: (Np x 2) double  coordinates
            - facets (optional): boundary facet vertex pairs (1-based)
            - facet_indices, facet_values (optional): tag structure
        """
        tdim = mesh.topology.dim

        mesh.topology.create_connectivity(tdim, 0)
        conn = mesh.topology.connectivity(tdim, 0).array
        num_cells = len(conn) // 3
        cells = conn.reshape(num_cells, 3) + 1  # MATLAB uses 1-based indexing

        points = mesh.geometry.x[:, :2]

        save_dict = {
            "cells": cells.astype(np.int32),
            "points": points.astype(float)
        }

        if facet_tags is not None:
            # Connectivity: facets → vertices
            mesh.topology.create_connectivity(1, 0)
            f2v = mesh.topology.connectivity(1, 0)

            # From adjacent list to Nx2 array
            arr = f2v.array
            offs = f2v.offsets
            vert_lens = offs[1:] - offs[:-1]
            if not np.all(vert_lens == 2):
                bad = np.where(vert_lens != 2)[0]
                raise RuntimeError(
                    f"Facet(s) {bad} have {vert_lens[bad]} vertices, expected two. "
                    "Mesh is not a pure triangular 2D mesh."
                )           
            
            facets = np.column_stack((arr[offs[:-1]], arr[offs[:-1] + 1])) + 1   # 1-based

            save_dict["facets"] = facets
            save_dict["facet_indices"] = facet_tags.indices + 1
            save_dict["facet_values"] = facet_tags.values

        savemat(filename, save_dict) 
        
        if facet_tags is not None:
            print(f"[export_domain] Saved MATLAB domain and facet tags: {filename})")
        else:
            print(f"[export_domain] Wrote MATLAB domain file: {filename}")
