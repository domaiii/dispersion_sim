import ufl
import warnings
import adios4dolfinx
import numpy as np
import pyvista as pv

from mpi4py import MPI
from pathlib import Path
from ufl import grad, div, dot, inner, dx
from basix.ufl import element, mixed_element
from dolfinx import fem, plot, mesh
from dolfinx.fem.petsc import assemble_vector, assemble_matrix
from petsc4py import PETSc

class Visualizer2D:

    def __init__(self, function_space: fem.FunctionSpace, window_size=(1600, 900), font_size=16):
        self.function_space = function_space
        self.topology, self.cell_type, self.geom = plot.vtk_mesh(function_space)
        self.grid = pv.UnstructuredGrid(self.topology, self.cell_type, self.geom)

        self.plotter = pv.Plotter(window_size=window_size)
        self._configure_style(font_size)

    def _configure_style(self, font_size):
        """Setzt einheitliche Fonts und Colorbar-Stil."""
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

    def add_scalar_field(self, name: str, scalar_func: fem.Function, cmap: str = "viridis"):
        value_size = scalar_func.x.block_size
        if value_size != 1:
            raise ValueError(f"{name} is no scalar field (value_size={value_size}) — ignored.")

        self.grid.point_data[name] = scalar_func.x.array
        self.plotter.add_mesh(
            self.grid.copy(),
            scalars=name,
            cmap=cmap,
            scalar_bar_args={**self.scalar_bar_args, "title": name},
        )

    def add_vector_field(self, name: str, vector_func: fem.Function, factor: float = 0.15):
        vec2d = vector_func.x.array.reshape(-1, 2)
        vec3d = np.hstack((vec2d, np.zeros((vec2d.shape[0], 1))))
        self.grid.point_data[name] = vec3d

        subset = self.grid.extract_points(np.arange(self.grid.n_points))
        glyphs = subset.glyph(orient=name, scale=name, factor=factor)
        self.plotter.add_mesh(glyphs, cmap="viridis", scalar_bar_args={**self.scalar_bar_args, "title": name})

    def add_points(self, coords, color="red", size=10, label="Measurements"):
        pts = pv.PolyData(coords)
        self.plotter.add_mesh(pts, color=color, point_size=size, label=label)

    def add_background_mesh(self, opacity=0.3, gridlines=False):
        self.plotter.add_mesh(self.grid, color="gray", opacity=opacity, show_edges=gridlines)

    def show(self, title=None, zoom=1.0):
        self.plotter.view_xy()
        self.plotter.add_axes()
        if title:
            self.plotter.add_text(title, position="upper_edge", font_size=16, color="black")
        self.plotter.zoom_camera(zoom)
        self.plotter.show()


class AirflowEstimator:
    def __init__(self,
                 domain: mesh,
                 w_measured: fem.Function,
                 measurement_ids_W,
                 kin_viscosity: float = 1.5e-4,
                 weight_misfit: float = 1e3,
                 weight_pde_error: float = 1e2,
                 weight_reg: float = 1e-3):
        """
        Initialisiert den Estimator OHNE Boundary Conditions.
        Alle Funktionsräume werden aufgebaut, damit man sie direkt für BCs nutzen kann.
        """
        self.domain = domain
        self.measurement_ids_W = np.asarray(measurement_ids_W, dtype=np.int32)
        self.w_measured = w_measured

        # Create spaces
        (self.W, self.W0, self.W1,
         self.V, self.Q,
         self.V_to_W, self.Q_to_W) = self.build_mixed_space(domain)
        
        self.domain.topology.create_connectivity(domain.topology.dim - 1, domain.topology.dim)
        
        # Parameters
        self.viscosity = kin_viscosity
        self.weight_misfit = weight_misfit
        self.weight_pde_error = weight_pde_error
        self.weight_reg = weight_reg

        # Optionals
        self.bcs: list[fem.DirichletBC] = []
        self.facet_tags = None
        self.w_final: fem.Function | None = None
        self.ground_truth: fem.Function | None = None

    @classmethod
    def from_file(cls,
                  bp_path: Path,
                  fun_name: str | None = "velocity",
                  meshtags_name: str | None = "facet_tags",
                  p: int = 100,
                  seed: int = 5):
        
        domain = adios4dolfinx.read_mesh(bp_path, MPI.COMM_WORLD)
        W, W0, W1, V, Q, V_to_W, Q_to_W = cls.build_mixed_space(domain)

        u_true = fem.Function(V)
        adios4dolfinx.read_function(bp_path, u_true, name=fun_name)

        w_true = fem.Function(W)
        w_true.sub(0).interpolate(u_true)

        coords_P2 = V.tabulate_dof_coordinates()
        rng = np.random.default_rng(seed)
        sample_ids = rng.choice(len(coords_P2), size=p, replace=False)
        x_ids, y_ids = sample_ids * 2, sample_ids * 2 + 1
        velocity_ids_V = np.stack((x_ids, y_ids)).T.flatten()
        measurement_ids_W = np.asarray(V_to_W, dtype=np.int32)[velocity_ids_V]

        w_measured = fem.Function(W)
        w_measured.x.array[:] = 0.0
        w_measured.x.array[measurement_ids_W] = w_true.x.array[measurement_ids_W]

        est = cls(domain, w_measured, measurement_ids_W)
        est.set_ground_truth(w_true)

        try:
            tags = adios4dolfinx.read_meshtags(bp_path, domain, meshtags_name)
        except RuntimeError as e:
            tags = None

        est.facet_tags = tags
    
        return est

    @staticmethod
    def build_mixed_space(domain, deg_u=2, deg_p=1):
        """Erzeugt das gemischte (velocity-pressure) Funktionsraumtuple."""
        elem_u = element("Lagrange", domain.basix_cell(), deg_u, shape=(domain.geometry.dim,))
        elem_p = element("Lagrange", domain.basix_cell(), deg_p)
        mixed_elem = mixed_element([elem_u, elem_p])

        W = fem.functionspace(domain, mixed_elem)
        W0, W1 = W.sub(0), W.sub(1)
        V, V_to_W = W0.collapse()
        Q, Q_to_W = W1.collapse()
        return W, W0, W1, V, Q, np.array(V_to_W, dtype=np.int32), np.array(Q_to_W, dtype=np.int32)


    def solve(self, maxit: int = 10, tol: float = 1e-2, damping: float | None = None):
        if not self.bcs:
            raise ValueError("No boundary conditions set. Use add_dirichlet_bc() to add BCs.")

        W = self.W
        wh = fem.Function(W)
        wh_prev = fem.Function(W)

        for k in range(maxit):
            A, b = self._build_linear_system(wh_prev)

            ksp = PETSc.KSP().create(A.comm)
            ksp.setOperators(A)
            ksp.setType("preonly")
            ksp.getPC().setType("lu")
            ksp.setFromOptions()
            ksp.solve(b, wh.x.petsc_vec)
            wh.x.petsc_vec.assemblyBegin(); wh.x.petsc_vec.assemblyEnd()
            wh.x.array[:] = wh.x.petsc_vec.getArray(readonly=True)

            step_norm = np.linalg.norm(wh_prev.x.array - wh.x.array)
            #print(f"Iter {k}: step_norm = {step_norm:.3e}")
            if step_norm < tol:
                #print(f"Converged after {k+1} iterations.")
                break

            if damping is not None and 0.0 < damping < 1.0:
                wh_prev.x.array[:] = (1 - damping) * wh_prev.x.array + damping * wh.x.array
            else:
                wh_prev.x.array[:] = wh.x.array

        self.w_final = wh
        return self.w_final
    
    def add_dirichlet_bc(self, bc: fem.DirichletBC | list[fem.DirichletBC]):
        if isinstance(bc, list):
            self.bcs += bc
        else:
            self.bcs.append(bc)

    def set_ground_truth(self, funW: fem.Function):
        if self.ground_truth:
            warnings.warn("Overwriting ground truth data.")
        else:
            self.ground_truth = funW
            
    def set_weights(self, kin_v: float | None, 
                          misfit: float | None, 
                          pde_err: float | None, 
                          reg: float | None ):
        if kin_v:   self.viscosity = kin_v
        if misfit:  self.weight_misfit = misfit
        if pde_err: self.weight_pde_error = pde_err
        if reg:     self.weight_reg = reg
        
    def get_measurement_coordinates(self) -> np.ndarray:
        """Rekonstruiere Messpunkt-Koordinaten aus measurement_ids_W."""
        coords_P2 = self.V.tabulate_dof_coordinates()
        W_to_V = {w: v for v, w in enumerate(self.V_to_W)}
        measured_v_ids = [W_to_V[i] for i in self.measurement_ids_W if i in W_to_V]
        measured_v_ids_unique = np.unique(np.array(measured_v_ids) // self.domain.geometry.dim)
        return coords_P2[measured_v_ids_unique]

    def _build_linear_system(self, wh_prev: fem.Function):
        """Baut das gesamte lineare System (A, b) inkl. PDE, Regularisierung und Daten-Misfit."""

        W = wh_prev.function_space
        domain = W.mesh

        # Konstanten
        nu    = fem.Constant(domain, PETSc.ScalarType(self.viscosity))
        beta  = fem.Constant(domain, PETSc.ScalarType(self.weight_pde_error))
        gamma = fem.Constant(domain, PETSc.ScalarType(self.weight_reg))

        # Test-/Trial-Funktionen
        uh_prev, _ = wh_prev.split()
        (u, p) = ufl.TrialFunctions(W)
        (v, q) = ufl.TestFunctions(W)

        # PDE Residuen (Least-Squares)
        Rmom_u = -nu * div(grad(u)) + dot(uh_prev, grad(u)) + grad(p)
        Rmom_v = -nu * div(grad(v)) + dot(uh_prev, grad(v)) + grad(q)
        Rdiv_u = div(u)
        Rdiv_v = div(v)

        a_pde = (beta * (inner(Rmom_u, Rmom_v) + Rdiv_u * Rdiv_v)) * dx
        a_reg = (gamma * inner(grad(u), grad(v))) * dx # regularization with ||grad(u)||
        #a_reg = (gamma * inner(u, v)) * dx # regularization with ||u||

        zero_vec = fem.Constant(domain, PETSc.ScalarType((0.0,) * domain.geometry.dim))
        L = inner(zero_vec, v) * dx

        # --- Assemble mit BCs
        aF, LF = fem.form(a_pde + a_reg), fem.form(L)
        A = assemble_matrix(aF, bcs=self.bcs); A.assemble()
        b = assemble_vector(LF)
        fem.apply_lifting(b, [aF], bcs=[self.bcs])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        fem.set_bc(b, self.bcs)

        # --- DOF-Penalty (Messdaten)
        S = PETSc.Mat().createAIJ(A.getSizes(), nnz=1, comm=A.comm); S.setUp()
        for i in map(int, self.measurement_ids_W):
            S.setValue(i, i, 1.0)
        S.assemble()

        rhs_add = self.w_measured.x.petsc_vec.duplicate()
        S.mult(self.w_measured.x.petsc_vec, rhs_add)

        A.axpy(self.weight_misfit, S, structure=PETSc.Mat.Structure.DIFFERENT_NONZERO_PATTERN)
        b.axpy(self.weight_misfit, rhs_add)
        return A, b
