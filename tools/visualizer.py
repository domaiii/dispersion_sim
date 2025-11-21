import numpy as np
import pyvista as pv

from dolfinx import fem, plot

class Visualizer2D:

    def __init__(self, function_space: fem.FunctionSpace, window_size=(1600, 900), font_size=16):
        self.function_space = function_space
        self.topology, self.cell_type, self.geom = plot.vtk_mesh(function_space)
        self.grid = pv.UnstructuredGrid(self.topology, self.cell_type, self.geom)

        self.plotter = pv.Plotter(window_size=window_size)
        
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

        subset = self.grid.extract_points(np.arange(self.grid.n_points))
        glyphs = subset.glyph(orient=name, scale=name, factor=factor)
        
        self.plotter.add_mesh(glyphs, cmap="coolwarm", 
                              scalar_bar_args={**self.scalar_bar_args, "title": f"{name} Magnitude"})

    def add_points(self, coords, 
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

    def show(self, title: str = None, zoom: float = 1.0):
        """
        Displays the plot.
        
        FIX: Changed 'self.plotter.mesh_actors' to check for any actors 
        in 'self.plotter.actors', which is the currently supported way in modern PyVista.
        """
        self.plotter.view_xy()
        self.plotter.add_axes()
        
        # Add legend
        if self.plotter.actors:
            self.plotter.add_legend(face="circle", size=(0.15, 0.1)) 
            
        if title:
            self.plotter.add_text(title, position="upper_edge", font_size=16, color="black")
        self.plotter.zoom_camera(zoom)
        self.plotter.show()