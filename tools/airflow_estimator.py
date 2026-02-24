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

class AirflowEstimator:
    def __init__(self,
                 domain: mesh.Mesh,
                 w_measured: fem.Function,
                 measurement_ids_W,
                 kin_viscosity: float = 1.5e-4,
                 weight_misfit: float = 1.0,
                 weight_pde_error: float = 1.0,
                 weight_reg: float = 1e-2):
        """
        Initialisiert den Estimator OHNE Boundary Conditions.
        Alle Funktionsräume werden aufgebaut, damit man sie direkt für BCs nutzen kann.
        """
        self.domain: mesh.Mesh = domain
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

        # Optional
        self.bcs: list[fem.DirichletBC] = []
        self.facet_tags = None
        self.w_final: fem.Function | None = None
        self.ground_truth: fem.Function | None = None

        self._boundary_name_to_id: dict[str, int] = {}

    @classmethod
    def from_file(cls,
                  bp_path: Path,
                  p: int | None = 0,
                  seed: int | None = 0,
                  fun_name: str | None = "velocity",
                  meshtags_name: str | None = "facet_tags",
                  meshfile: Path | None = None):
        """
        Construct estimator from ADIOS bp file.

        Parameters
        ----------
        bp_path : Path
            ADIOS .bp file containing mesh, velocity, facet tags.
        p : int
            Number of velocity measurement points.
        seed : int
            Random seed for measurement sampling.
        fun_name : str
            Name of the velocity function in the file.
        meshtags_name : str
            Name of facet tags in the file.
        meshfile : Path | None
            Optional Gmsh .msh file to recover physical group names
            (e.g. 'Walls', 'Outflow'). If provided, we build a
            name -> id mapping used by set_*_bc helpers.
        """

        domain = adios4dolfinx.read_mesh(bp_path, MPI.COMM_WORLD)
        W, W0, W1, V, Q, V_to_W, Q_to_W = cls.build_mixed_space(domain)

        # Ground truth velocity in V
        u_true = fem.Function(V)
        adios4dolfinx.read_function(bp_path, u_true, name=fun_name)

        # Embed into mixed space W as (u_true, p=0)
        w_true = fem.Function(W)
        w_true.sub(0).interpolate(u_true)

        # Random sampling of velocity DOFs in V
        coords_P2 = V.tabulate_dof_coordinates()
        rng = np.random.default_rng(seed)
        sample_ids = rng.choice(len(coords_P2), size=p, replace=False)
        x_ids, y_ids = sample_ids * 2, sample_ids * 2 + 1
        velocity_ids_V = np.stack((x_ids, y_ids)).T.flatten()
        measurement_ids_W = np.asarray(V_to_W, dtype=np.int32)[velocity_ids_V]

        # Measured mixed field
        w_measured = fem.Function(W)
        w_measured.x.array[:] = 0.0
        w_measured.x.array[measurement_ids_W] = w_true.x.array[measurement_ids_W]

        # Build estimator
        est = cls(domain, w_measured, measurement_ids_W)
        est.set_ground_truth(w_true)

        # Read facet tags from bp (if available)
        if meshtags_name is not None:
            try:
                tags = adios4dolfinx.read_meshtags(bp_path, domain, meshtags_name)
            except RuntimeError:
                tags = None
        else:
            tags = None

        est.facet_tags = tags

        # Optional: read physical names from meshfile
        if meshfile is not None:
            import gmsh
            gmsh.initialize()
            gmsh.open(str(meshfile))
            phy_groups = gmsh.model.getPhysicalGroups()
            name_to_id = {
                gmsh.model.getPhysicalName(dim, tag): tag
                for (dim, tag) in phy_groups
            }
            gmsh.finalize()
            est._boundary_name_to_id = name_to_id
        else:
            est._boundary_name_to_id = {}

        return est
    
    def _ensure_boundary_name_map(self):
        if self.facet_tags is None:
            raise RuntimeError("facet_tags is not set. Make sure from_file "
                               "was called with a valid meshtags_name.")
        if not hasattr(self, "_boundary_name_to_id") or not self._boundary_name_to_id:
            raise RuntimeError(
                "No boundary name->id mapping available. "
                "Call from_file(..., meshfile=...) or set _boundary_name_to_id manually."
            )

    def set_no_slip_bc(self, wall_names: str | list[str]):
        """
        Apply no-slip (u=0) boundary condition on the given physical boundaries.

        Parameters
        ----------
        wall_names : str or list[str]
            Physical group names, e.g. 'Walls' or ['Walls', 'Obstacles'].
        """
        self._ensure_boundary_name_map()

        if isinstance(wall_names, str):
            names = [wall_names]
        else:
            names = list(wall_names)

        # collect facet indices for all given names
        domain = self.domain
        ft = self.facet_tags
        import numpy as np

        facets = np.concatenate([
            ft.find(self._boundary_name_to_id[name]) for name in names
        ])

        # u = 0 on these facets
        u_D = fem.Function(self.V)
        u_D.x.array[:] = 0.0

        dofs = fem.locate_dofs_topological((self.W0, self.V),
                                           domain.topology.dim - 1,
                                           facets)
        bc = fem.dirichletbc(u_D, dofs, self.W0)
        self.add_dirichlet_bc(bc)
        return bc

    def set_zero_pressure_bc(self, outlet_names: str | list[str]):
        """
        Apply p=0 boundary condition on the given outlet boundaries.

        Parameters
        ----------
        outlet_names : str or list[str]
            Physical group names, e.g. 'Outflow'.
        """
        self._ensure_boundary_name_map()

        if isinstance(outlet_names, str):
            names = [outlet_names]
        else:
            names = list(outlet_names)

        domain = self.domain
        ft = self.facet_tags
        import numpy as np

        facets = np.concatenate([
            ft.find(self._boundary_name_to_id[name]) for name in names
        ])

        p_zero = fem.Function(self.Q)
        p_zero.x.array[:] = 0.0

        dofs = fem.locate_dofs_topological((self.W1, self.Q),
                                           domain.topology.dim - 1,
                                           facets)
        bc = fem.dirichletbc(p_zero, dofs, self.W1)
        self.add_dirichlet_bc(bc)
        return bc
        

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

    def reset_random_measurements(self, p: int, seed: int | None = None):
        """
        Creates new random measurement set overwriting self.measurement_ids_W and selfw_measured.
        Resets the solution self.w_final.
        Benötigt, dass eine ground_truth-Funktion gesetzt wurde.
        """
        if self.ground_truth is None:
            raise ValueError("No ground truth set. Use set_ground_truth() first.")

        rng = np.random.default_rng(seed)

        coords_V = self.V.tabulate_dof_coordinates()
        ndofs = len(coords_V)

        sample_ids = rng.choice(ndofs, size=p, replace=False)

        x_ids = sample_ids * 2
        y_ids = sample_ids * 2 + 1
        velocity_ids_V = np.stack((x_ids, y_ids)).T.flatten()

        # Mapping in W
        measurement_ids_W = self.V_to_W[velocity_ids_V]

        self.w_measured.x.array[:] = 0.0
        self.w_measured.x.array[measurement_ids_W] = \
            self.ground_truth.x.array[measurement_ids_W]

        self.measurement_ids_W = measurement_ids_W.astype(np.int32)
        self.w_final = None

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
