import ufl
import warnings
import adios4dolfinx
import numpy as np
import pandas as pd

from mpi4py import MPI
from pathlib import Path
from scipy.spatial import cKDTree
from typing import Callable
from ufl import grad, div, dot, inner, dx
from basix.ufl import element, mixed_element
from dolfinx import fem, mesh
from dolfinx.fem.petsc import assemble_vector, assemble_matrix
from petsc4py import PETSc
import scipy.sparse as sps
from scipy.sparse.linalg import lsqr

class AirflowEstimator:

    def __init__(self,
                 domain: mesh.Mesh,
                 w_measured: fem.Function,
                 measurement_ids_W,
                 kin_viscosity: float = 1.5e-4,
                 weight_misfit: float = 1.0,
                 weight_pde_error: float = 1.0,
                 weight_reg: float = 1e-2,
                 weight_boundary: float = 1000.0,
                 regularization_mode: str = "smooth"):
        """
        Initialisiert den Estimator OHNE Boundary Conditions.
        Alle Funktionsräume werden aufgebaut, damit man sie direkt für BCs nutzen kann.
        """
        self.domain: mesh.Mesh = domain
        self.measurement_ids_W = np.asarray(measurement_ids_W, dtype=np.int32)
        self.w_measured = w_measured

        (self.W, self.W0, self.W1,
         self.V, self.Q,
         self.V_to_W, self.Q_to_W) = self.build_mixed_space(domain)
        
        self.domain.topology.create_connectivity(domain.topology.dim - 1, domain.topology.dim)
        
        self.viscosity = kin_viscosity
        self.weight_misfit = weight_misfit
        self.weight_pde_error = weight_pde_error
        self.weight_reg = weight_reg
        self.weight_boundary = weight_boundary
        self.regularization_mode = self._normalize_regularization_mode(regularization_mode)

        self.bcs: list[fem.DirichletBC] = []
        self.facet_tags = None
        self.w_final: fem.Function | None = None
        self.ground_truth: fem.Function | None = None
        self._smooth_lsq_operator: sps.csr_matrix | None = None

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

        if meshtags_name is not None:
            try:
                tags = adios4dolfinx.read_meshtags(bp_path, domain, meshtags_name)
            except RuntimeError:
                tags = None
        else:
            tags = None

        est.facet_tags = tags

        if meshfile is not None:
            est._boundary_name_to_id = cls._read_physical_name_map(meshfile)
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

        facets = np.concatenate([
            self.facet_tags.find(self._boundary_name_to_id[name]) for name in names
        ])

        u_D = fem.Function(self.V)
        u_D.x.array[:] = 0.0

        dofs = fem.locate_dofs_topological((self.W0, self.V),
                                           self.domain.topology.dim - 1,
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

        facets = np.concatenate([
            self.facet_tags.find(self._boundary_name_to_id[name]) for name in names
        ])

        p_zero = fem.Function(self.Q)
        p_zero.x.array[:] = 0.0

        dofs = fem.locate_dofs_topological((self.W1, self.Q),
                                           self.domain.topology.dim - 1,
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

    @staticmethod
    def _num_dofs(space) -> int:
        return space.dofmap.index_map.size_global * space.dofmap.index_map_bs


    def _solve_fixed_point(self,
                           system_generator: Callable[[fem.Function], tuple[PETSc.Mat, PETSc.Vec]],
                           maxit: int = 10,
                           tol: float = 1e-2,
                           damping: float | None = None):
        if not self.bcs:
            raise ValueError("No boundary conditions set. Use add_dirichlet_bc() to add BCs.")

        W = self.W
        wh = fem.Function(W)
        wh_prev = fem.Function(W)

        for k in range(maxit):
            A, b = system_generator(wh_prev)

            ksp = PETSc.KSP().create(A.comm)
            ksp.setOperators(A)
            ksp.setType("preonly")
            ksp.getPC().setType("lu")
            ksp.setFromOptions()
            ksp.solve(b, wh.x.petsc_vec)
            wh.x.petsc_vec.assemblyBegin(); wh.x.petsc_vec.assemblyEnd()
            wh.x.array[:] = wh.x.petsc_vec.getArray(readonly=True)

            step_norm = np.linalg.norm(wh_prev.x.array - wh.x.array)
            if step_norm < tol:
                break

            if damping is not None and 0.0 < damping < 1.0:
                wh_prev.x.array[:] = (1 - damping) * wh_prev.x.array + damping * wh.x.array
            else:
                wh_prev.x.array[:] = wh.x.array

        self.w_final = wh
        return self.w_final

    @staticmethod
    def _normalize_regularization_mode(mode: str) -> str:
        mode_norm = mode.strip().lower()
        if mode_norm not in {"smooth", "value"}:
            raise ValueError("regularization mode must be 'smooth' or 'value'.")
        return mode_norm

    def set_regularization(self, mode: str):
        self.regularization_mode = self._normalize_regularization_mode(mode)

    def _resolve_regularization_mode(self, mode: str | None = None) -> str:
        if mode is None:
            return self.regularization_mode
        return self._normalize_regularization_mode(mode)

    def solve_minimum_residual(self,
                               maxit: int = 10,
                               tol: float = 1e-2,
                               damping: float | None = None,
                               regularization: str | None = None):
        """Solve via direct minimum-residual formulation with strong Dirichlet BCs."""
        reg_mode = self._resolve_regularization_mode(regularization)
        return self._solve_fixed_point(
            system_generator=lambda wh_prev: self._build_min_residual_system(wh_prev, reg_mode=reg_mode),
            maxit=maxit,
            tol=tol,
            damping=damping,
        )

    def solve_weak_penalty(self,
                           maxit: int = 10,
                           tol: float = 1e-2,
                           damping: float | None = None,
                           regularization: str | None = None):
        """Solve via weak-form Galerkin system with penalty terms for BCs and measurements."""
        reg_mode = self._resolve_regularization_mode(regularization)
        return self._solve_fixed_point(
            system_generator=lambda wh_prev: self._build_weak_form_system_penalty(wh_prev, reg_mode=reg_mode),
            maxit=maxit,
            tol=tol,
            damping=damping,
        )

    def solve_linear_least_squares(self,
                                   maxit: int = 10,
                                   tol: float = 1e-3,
                                   regularization: str | None = None):
        """Solve the stacked weak-form system as an overdetermined linear least-squares problem."""
        reg_mode = self._resolve_regularization_mode(regularization)
        wh = fem.Function(self.W)
        wh_prev = fem.Function(self.W)
        wh_prev.x.array[:] = 0.0

        num_total_dofs = self._num_dofs(self.W)
        reg_op = self._build_linear_regularization_operator(reg_mode=reg_mode)

        for k in range(maxit):
            K_petsc, f_petsc = self._build_weak_form_system(wh_prev)
            ai, aj, av = K_petsc.getValuesCSR()
            K_sp = sps.csr_matrix((av, aj, ai), shape=(num_total_dofs, num_total_dofs))
            f_np = f_petsc.array.reshape(-1, 1)

            bc_rows, bc_cols, bc_vals, bc_rhs = [], [], [], []
            for bc in self.bcs:
                for dof in bc.dof_indices()[0]:
                    bc_rows.append(len(bc_rows))
                    bc_cols.append(dof)
                    bc_vals.append(1.0)
                    bc_rhs.append(0.0)
            R_sp = sps.csr_matrix((bc_vals, (bc_rows, bc_cols)), shape=(len(bc_rows), num_total_dofs))
            r_np = np.zeros((len(bc_rows), 1))

            m_idx = self.measurement_ids_W
            m_val = self.w_measured.x.array[m_idx].reshape(-1, 1)
            M_sp = sps.csr_matrix((np.ones_like(m_idx, dtype=float), (np.arange(len(m_idx)), m_idx)),
                                  shape=(len(m_idx), num_total_dofs))

            A_stack = sps.vstack([
                np.sqrt(self.weight_pde_error) * K_sp,
                np.sqrt(self.weight_boundary) * R_sp,
                np.sqrt(self.weight_misfit) * M_sp,
                np.sqrt(self.weight_reg) * reg_op,
            ]).tocsr()

            zeros_reg = np.zeros((reg_op.shape[0], 1))
            b_stack = np.vstack([
                -np.sqrt(self.weight_pde_error) * f_np,
                np.sqrt(self.weight_boundary) * r_np,
                np.sqrt(self.weight_misfit) * m_val,
                zeros_reg,
            ]).reshape(-1)

            res = lsqr(A_stack, b_stack, iter_lim=5000)[0]
            wh.x.array[:] = res

            diff = np.linalg.norm(wh.x.array - wh_prev.x.array) / (np.linalg.norm(wh.x.array) + 1e-10)
            print(f"Iteration {k}: Rel. Error = {diff:.2e}")

            if diff < tol:
                break
            wh_prev.x.array[:] = wh.x.array

        self.w_final = wh
        return wh

    def add_dirichlet_bc(self, bc: fem.DirichletBC | list[fem.DirichletBC]):
        if isinstance(bc, list):
            self.bcs += bc
        else:
            self.bcs.append(bc)

    def set_measurements(
        self,
        measurement_ids_W: np.ndarray,
        measurement_values: np.ndarray,
        clear_existing: bool = True,
    ):
        """
        Set explicit wind measurements in mixed space W.

        Parameters
        ----------
        measurement_ids_W : np.ndarray
            Flattened W-indices for velocity components.
        measurement_values : np.ndarray
            Flattened measurement values aligned with measurement_ids_W.
        clear_existing : bool
            If True, clear all previous measurements first.
        """
        ids = np.asarray(measurement_ids_W, dtype=np.int32).reshape(-1)
        values = np.asarray(measurement_values, dtype=float).reshape(-1)

        if ids.size == 0:
            raise ValueError("measurement_ids_W is empty.")
        if ids.size != values.size:
            raise ValueError(
                f"Length mismatch: len(ids)={ids.size} != len(values)={values.size}"
            )
        if np.any(ids < 0) or np.any(ids >= self.w_measured.x.array.size):
            raise ValueError("measurement_ids_W contains out-of-bounds indices.")

        if clear_existing:
            self.w_measured.x.array[:] = 0.0

        self.w_measured.x.array[ids] = values
        self.measurement_ids_W = ids
        self.w_final = None
        self._smooth_lsq_operator = None

    def set_measurements_from_csv(
        self,
        samples_csv: str | Path,
        unique_nodes: bool = True,
        max_xy_dist: float | None = None,
        clear_existing: bool = True,
    ) -> dict[str, float]:
        """
        Load wind samples from CSV and map them to nearest velocity nodes.

        Expected CSV columns: x, y, wind_x, wind_y.

        Parameters
        ----------
        samples_csv : str | Path
            Path to sample CSV.
        unique_nodes : bool
            If True, keep only first sample per matched FEM node.
        max_xy_dist : float | None
            Optional maximum allowed nearest-neighbor mapping distance in XY.
        clear_existing : bool
            If True, remove all previous measurements before writing new ones.

        Returns
        -------
        dict[str, float]
            Mapping stats (input/used samples, dropped duplicates, max distance).
        """
        samples_csv = Path(samples_csv).resolve(strict=True)

        df = pd.read_csv(samples_csv)
        required = ["x", "y", "wind_x", "wind_y"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(
                f"Missing required columns in {samples_csv.name}: {missing}. "
                f"Expected at least {required}."
            )
        if len(df) == 0:
            raise ValueError(f"No sample rows found in {samples_csv}")

        samples_xy = df[["x", "y"]].to_numpy(dtype=float)
        samples_uv = df[["wind_x", "wind_y"]].to_numpy(dtype=float)

        node_xy = np.asarray(self.V.tabulate_dof_coordinates(), dtype=float)[:, :2]
        tree = cKDTree(node_xy)
        dist, node_ids = tree.query(samples_xy, k=1, p=2.0, workers=-1)

        max_dist = float(np.max(dist))
        if max_xy_dist is not None and max_dist > max_xy_dist:
            raise ValueError(
                f"Maximum XY mapping distance exceeded: {max_dist:.6g} > {max_xy_dist:.6g}"
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
        measurement_ids_W = self.V_to_W[velocity_ids_V]
        measurement_values = np.stack((samples_uv[:, 0], samples_uv[:, 1]), axis=1).flatten()

        self.set_measurements(
            measurement_ids_W=measurement_ids_W,
            measurement_values=measurement_values,
            clear_existing=clear_existing,
        )

        return {
            "n_input_samples": float(n_input),
            "n_used_samples": float(len(node_ids)),
            "n_dropped_duplicate_nodes": float(n_dropped),
            "max_xy_dist": float(np.max(dist)) if len(dist) else 0.0,
        }

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

        measurement_ids_W = self.V_to_W[velocity_ids_V]
        measurement_values = self.ground_truth.x.array[measurement_ids_W]
        self.set_measurements(
            measurement_ids_W=measurement_ids_W,
            measurement_values=measurement_values,
            clear_existing=True,
        )

    def set_ground_truth(self, funW: fem.Function):
        if self.ground_truth:
            warnings.warn("Overwriting ground truth data.")
        else:
            self.ground_truth = funW
            
    def set_weights(self, kin_v: float | None = None, 
                          misfit: float | None = None, 
                          pde_err: float | None = None, 
                          reg: float | None = None,
                          boundary: float | None = None):
        if kin_v is not None:
            self.viscosity = kin_v
        if misfit is not None:
            self.weight_misfit = misfit
        if pde_err is not None:
            self.weight_pde_error = pde_err
        if reg is not None:
            self.weight_reg = reg
        if boundary is not None:
            self.weight_boundary = boundary
        
    def get_measurement_coordinates(self) -> np.ndarray:
        """Rekonstruiere Messpunkt-Koordinaten aus measurement_ids_W."""
        coords_P2 = self.V.tabulate_dof_coordinates()
        W_to_V = {w: v for v, w in enumerate(self.V_to_W)}
        measured_v_ids = [W_to_V[i] for i in self.measurement_ids_W if i in W_to_V]
        measured_v_ids_unique = np.unique(np.array(measured_v_ids) // self.domain.geometry.dim)
        return coords_P2[measured_v_ids_unique]

    def evaluate_objective_terms(self,
                                 wh: fem.Function | None = None,
                                 regularization: str | None = None) -> dict[str, float]:
        """
        Evaluate objective contributions for a given mixed field `wh`.
        Returns unweighted and weighted values of PDE error, regularization, and data misfit.
        """
        if wh is None:
            if self.w_final is None:
                raise ValueError("No solution available. Call a solve_* method first or provide `wh`.")
            wh = self.w_final

        domain = self.domain
        nu = fem.Constant(domain, PETSc.ScalarType(self.viscosity))
        beta = float(self.weight_pde_error)
        gamma = float(self.weight_reg)
        alpha = float(self.weight_misfit)
        reg_mode = self._resolve_regularization_mode(regularization)

        uh, ph = ufl.split(wh)
        Rmom = -nu * div(grad(uh)) + dot(uh, grad(uh)) + grad(ph)
        Rdiv = div(uh)

        pde_unweighted_form = fem.form((inner(Rmom, Rmom) + Rdiv * Rdiv) * dx)
        if reg_mode == "value":
            reg_unweighted_form = fem.form(inner(uh, uh) * dx)
        else:
            reg_unweighted_form = fem.form(inner(grad(uh), grad(uh)) * dx)

        pde_unweighted_local = fem.assemble_scalar(pde_unweighted_form)
        reg_unweighted_local = fem.assemble_scalar(reg_unweighted_form)

        pde_unweighted = domain.comm.allreduce(pde_unweighted_local, op=MPI.SUM)
        reg_unweighted = domain.comm.allreduce(reg_unweighted_local, op=MPI.SUM)

        diff = wh.x.array - self.w_measured.x.array
        local_ids = self.measurement_ids_W[
            (self.measurement_ids_W >= 0) & (self.measurement_ids_W < diff.size)
        ]
        misfit_unweighted_local = np.sum(diff[local_ids] ** 2)
        misfit_unweighted = domain.comm.allreduce(misfit_unweighted_local, op=MPI.SUM)

        pde_weighted = beta * pde_unweighted
        reg_weighted = gamma * reg_unweighted
        misfit_weighted = alpha * misfit_unweighted

        return {
            "pde_unweighted": float(pde_unweighted),
            "reg_unweighted": float(reg_unweighted),
            "misfit_unweighted": float(misfit_unweighted),
            "pde_weighted": float(pde_weighted),
            "reg_weighted": float(reg_weighted),
            "misfit_weighted": float(misfit_weighted),
            "objective_total_weighted": float(pde_weighted + reg_weighted + misfit_weighted),
        }

    def _build_min_residual_system(self, wh_prev: fem.Function, reg_mode: str | None = None):
        """Baut das gesamte lineare System (A, b) inkl. PDE, Regularisierung und Daten-Misfit."""

        W = wh_prev.function_space
        domain = W.mesh
        reg_mode = self._resolve_regularization_mode(reg_mode)

        nu    = fem.Constant(domain, PETSc.ScalarType(self.viscosity))
        beta  = fem.Constant(domain, PETSc.ScalarType(self.weight_pde_error))
        gamma = fem.Constant(domain, PETSc.ScalarType(self.weight_reg))

        uh_prev, _ = wh_prev.split()
        (u, p) = ufl.TrialFunctions(W)
        (v, q) = ufl.TestFunctions(W)

        Rmom_u = -nu * div(grad(u)) + dot(uh_prev, grad(u)) + grad(p)
        Rmom_v = -nu * div(grad(v)) + dot(uh_prev, grad(v)) + grad(q)
        Rdiv_u = div(u)
        Rdiv_v = div(v)

        a_pde = (beta * (inner(Rmom_u, Rmom_v) + Rdiv_u * Rdiv_v)) * dx
        if reg_mode == "value":
            a_reg = (gamma * inner(u, v)) * dx
        else:
            a_reg = (gamma * inner(grad(u), grad(v))) * dx

        zero_vec = fem.Constant(domain, PETSc.ScalarType((0.0,) * domain.geometry.dim))
        L = inner(zero_vec, v) * dx

        # Strong BC assembly
        aF, LF = fem.form(a_pde + a_reg), fem.form(L)
        A = assemble_matrix(aF, bcs=self.bcs); A.assemble()
        b = assemble_vector(LF)
        fem.apply_lifting(b, [aF], bcs=[self.bcs])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        fem.set_bc(b, self.bcs)

        # Measurement penalty
        S = PETSc.Mat().createAIJ(A.getSizes(), nnz=1, comm=A.comm); S.setUp()
        for i in map(int, self.measurement_ids_W):
            S.setValue(i, i, 1.0)
        S.assemble()

        rhs_add = self.w_measured.x.petsc_vec.duplicate()
        S.mult(self.w_measured.x.petsc_vec, rhs_add)

        A.axpy(self.weight_misfit, S, structure=PETSc.Mat.Structure.DIFFERENT_NONZERO_PATTERN)
        b.axpy(self.weight_misfit, rhs_add)
        return A, b

    def _build_weak_form_system(self, wh_prev: fem.Function):
        (u, p) = ufl.TrialFunctions(self.W)
        (v, q) = ufl.TestFunctions(self.W)
        uh_prev, _ = ufl.split(wh_prev)

        nu = fem.Constant(self.domain, PETSc.ScalarType(self.viscosity))
        
        a_form = fem.form((
            inner(nu * grad(u), grad(v)) 
            + inner(grad(u) * uh_prev, v) 
            - p * div(v) + q * div(u)
        ) * dx)

        f_form = fem.form(inner(fem.Constant(self.domain, PETSc.ScalarType((0,0))), v) * dx)

        K = assemble_matrix(a_form)
        K.assemble()
        f = assemble_vector(f_form)
        
        return K, f
    
    def _build_weak_form_system_penalty(self, wh_prev: fem.Function, reg_mode: str | None = None):
        uh_prev, _ = wh_prev.split()
        (u, p) = ufl.TrialFunctions(self.W)
        (v, q) = ufl.TestFunctions(self.W)
        domain = self.domain
        reg_mode = self._resolve_regularization_mode(reg_mode)

        nu = fem.Constant(domain, PETSc.ScalarType(self.viscosity))
        w_pde = fem.Constant(domain, PETSc.ScalarType(self.weight_pde_error)) 
        w_reg = fem.Constant(domain, PETSc.ScalarType(self.weight_reg))

        if reg_mode == "value":
            reg_term = inner(u, v)
        else:
            reg_term = inner(grad(u), grad(v))

        a_form = fem.form((
            w_pde * (inner(nu * grad(u), grad(v)) 
            + inner(grad(u) * uh_prev, v) 
            - p * div(v) + q * div(u))
            + w_reg * reg_term
        ) * dx)

        zero_vec = fem.Constant(domain, PETSc.ScalarType((0.0,) * domain.geometry.dim))
        L_form = fem.form(inner(zero_vec, v) * dx)

        K_reg = assemble_matrix(a_form) 
        K_reg.assemble()
        f_reg = assemble_vector(L_form)

        # Boundary penalty
        R = PETSc.Mat().createAIJ(K_reg.getSizes(), nnz=1, comm=K_reg.comm)
        R.setUp()
        r_vec = f_reg.duplicate()
        r_vec.set(0.0)

        w_bc = self.weight_boundary

        for bc in self.bcs:
            dofs = bc.dof_indices()[0]
            for dof in dofs:
                R.setValue(dof, dof, 1.0)
        R.assemble()

        # Measurement penalty
        S = PETSc.Mat().createAIJ(K_reg.getSizes(), nnz=1, comm=K_reg.comm)
        S.setUp()
        for i in map(int, self.measurement_ids_W):
            S.setValue(i, i, 1.0)
        S.assemble()
        
        s_vec = self.w_measured.x.petsc_vec.duplicate()
        S.mult(self.w_measured.x.petsc_vec, s_vec)

        K_reg.axpy(w_bc, R, structure=PETSc.Mat.Structure.DIFFERENT_NONZERO_PATTERN)
        K_reg.axpy(self.weight_misfit, S, structure=PETSc.Mat.Structure.DIFFERENT_NONZERO_PATTERN)

        f_reg.axpy(w_bc, r_vec)
        f_reg.axpy(self.weight_misfit, s_vec)

        diag = K_reg.getDiagonal()
        if diag.min()[0] < 1e-15:
            print(f"WARNUNG: Matrix hat extrem kleine Diagonalelemente! Min: {diag.min()[0]}")
        if np.any(np.isnan(f_reg.array)):
            print("FEHLER: f_reg enthält NaNs!")
            
        return K_reg, f_reg

    def _build_linear_regularization_operator(self, reg_mode: str | None = None) -> sps.csr_matrix:
        reg_mode = self._resolve_regularization_mode(reg_mode)
        num_total_dofs = self._num_dofs(self.W)

        if reg_mode == "value":
            v_map = self.W.sub(0).dofmap
            u_dofs = np.unique(v_map.list.flatten())
            rows = np.arange(len(u_dofs))
            return sps.csr_matrix((np.ones_like(u_dofs, dtype=float), (rows, u_dofs)),
                                  shape=(len(u_dofs), num_total_dofs))

        if self._smooth_lsq_operator is not None:
            return self._smooth_lsq_operator

        coords = np.asarray(self.V.tabulate_dof_coordinates(), dtype=float)[:, :self.domain.geometry.dim]
        num_points = len(coords)
        if num_points < 2:
            self._smooth_lsq_operator = sps.csr_matrix((0, num_total_dofs))
            return self._smooth_lsq_operator

        k = min(5, num_points)
        tree = cKDTree(coords)
        dists, neighbors = tree.query(coords, k=k, p=2.0, workers=-1)

        rows, cols, vals = [], [], []
        row_id = 0
        seen_edges: set[tuple[int, int]] = set()
        dim = self.domain.geometry.dim

        for i in range(num_points):
            for dist_ij, j in zip(np.atleast_1d(dists[i])[1:], np.atleast_1d(neighbors[i])[1:]):
                edge = (i, int(j)) if i < int(j) else (int(j), i)
                if edge[0] == edge[1] or edge in seen_edges:
                    continue
                seen_edges.add(edge)
                weight = 1.0 / max(float(dist_ij), 1e-12)
                for comp in range(dim):
                    wi = int(self.V_to_W[i * dim + comp])
                    wj = int(self.V_to_W[int(j) * dim + comp])
                    rows.extend([row_id, row_id])
                    cols.extend([wi, wj])
                    vals.extend([weight, -weight])
                    row_id += 1

        self._smooth_lsq_operator = sps.csr_matrix((vals, (rows, cols)), shape=(row_id, num_total_dofs))
        return self._smooth_lsq_operator
