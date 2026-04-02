from dataclasses import dataclass
from abc import ABC, abstractmethod

import numpy as np
import scipy.sparse as sps
import ufl

from dolfinx import fem, mesh
from dolfinx.fem.petsc import assemble_matrix, assemble_vector
from mpi4py import MPI
from petsc4py import PETSc
from scipy.spatial import cKDTree
from scipy.sparse.linalg import lsqr
from ufl import div, dot, dx, grad, inner


@dataclass(slots=True)
class AirflowSolverConfig:
    domain: mesh.Mesh
    W: fem.FunctionSpace
    V: fem.FunctionSpace
    V_to_W: np.ndarray
    bcs: list[fem.DirichletBC]
    w_measured: fem.Function
    measurement_ids_W: np.ndarray
    viscosity: float
    weight_misfit: float
    weight_pde_res: float
    weight_reg: float
    weight_boundary: float
    regularization_mode: str


class BaseAirflowSolver(ABC):

    def __init__(self, ctx: AirflowSolverConfig):
        self.ctx = ctx

    def solve(self,
              maxit: int,
              tol: float,
              damping: float | None = None,
              regularization: str | None = None,
              verbose: bool = False):
        wh = fem.Function(self.ctx.W)
        wh_prev = fem.Function(self.ctx.W)
        wh_prev.x.array[:] = 0.0

        reg_mode = self._resolve_regularization_mode(regularization)
        self._validate()
        context = self._prepare_solve(reg_mode)

        for k in range(maxit):
            self._solve_step(wh_prev, wh, reg_mode, context)

            diff = np.linalg.norm(wh.x.array - wh_prev.x.array) / (np.linalg.norm(wh.x.array) + 1e-10)
            if verbose:
                self._report_iteration(k, diff, wh, reg_mode, context)
            if diff < tol:
                break

            if damping is not None and 0.0 < damping < 1.0:
                wh_prev.x.array[:] = (1 - damping) * wh_prev.x.array + damping * wh.x.array
            else:
                wh_prev.x.array[:] = wh.x.array

        if verbose:
            self._report_summary(wh, reg_mode, context)
        self._finalize_solve(context)
        return wh

    @staticmethod
    def normalize_regularization_mode(mode: str) -> str:
        mode_norm = mode.strip().lower()
        if mode_norm not in {"smooth", "value"}:
            raise ValueError("regularization mode must be 'smooth' or 'value'.")
        return mode_norm

    def _validate(self):
        if not self.ctx.bcs:
            raise ValueError("No boundary conditions set. Use add_dirichlet_bc() to add BCs.")

    def _resolve_regularization_mode(self, mode: str | None = None) -> str:
        if mode is None:
            return self.ctx.regularization_mode
        return self.normalize_regularization_mode(mode)

    def _prepare_solve(self, reg_mode: str):
        return None

    def _finalize_solve(self, context):
        return None

    @staticmethod
    def _num_dofs(space) -> int:
        return space.dofmap.index_map.size_global * space.dofmap.index_map_bs

    def _build_weak_form_system(self, wh_prev: fem.Function):
        (u, p) = ufl.TrialFunctions(self.ctx.W)
        (v, q) = ufl.TestFunctions(self.ctx.W)
        uh_prev, _ = ufl.split(wh_prev)

        nu = fem.Constant(self.ctx.domain, PETSc.ScalarType(self.ctx.viscosity))
        zero_vec = fem.Constant(
            self.ctx.domain,
            PETSc.ScalarType((0.0,) * self.ctx.domain.geometry.dim),
        )

        a_form = fem.form((
            inner(nu * grad(u), grad(v))
            + inner(grad(u) * uh_prev, v)
            - p * div(v)
            + q * div(u)
        ) * dx)
        f_form = fem.form(inner(zero_vec, v) * dx)

        K = assemble_matrix(a_form)
        K.assemble()
        f = assemble_vector(f_form)
        return K, f

    def _measurement_misfit(self, wh: fem.Function) -> float:
        diff = wh.x.array - self.ctx.w_measured.x.array
        m_idx = self.ctx.measurement_ids_W
        return float(np.sum(diff[m_idx] ** 2))

    def _boundary_penalty(self, wh: fem.Function) -> float:
        bc_dofs = []
        for bc in self.ctx.bcs:
            bc_dofs.extend(map(int, bc.dof_indices()[0]))
        if not bc_dofs:
            return 0.0
        values = wh.x.array[np.asarray(bc_dofs, dtype=np.int32)]
        return float(np.sum(values ** 2))

    def _value_regularization(self, wh: fem.Function, reg_mode: str) -> float:
        uh, _ = ufl.split(wh)
        if reg_mode == "value":
            form = fem.form(inner(uh, uh) * dx)
        else:
            form = fem.form(inner(grad(uh), grad(uh)) * dx)
        return float(self.ctx.domain.comm.allreduce(fem.assemble_scalar(form), op=MPI.SUM))

    def _weak_form_residual(self, wh: fem.Function) -> float:
        K, f = self._build_weak_form_system(wh)
        residual = f.duplicate()
        K.mult(wh.x.petsc_vec, residual)
        residual.axpy(-1.0, f)
        return float(residual.norm(PETSc.NormType.NORM_2) ** 2)

    @abstractmethod
    def _solve_step(self, wh_prev: fem.Function, wh: fem.Function, reg_mode: str, context):
        pass

    def _report_iteration(self, k: int, diff: float, wh: fem.Function, reg_mode: str, context):
        print(f"Iteration {k}: Rel. Error = {diff:.2e}")

    def _report_summary(self, wh: fem.Function, reg_mode: str, context):
        terms = self._evaluate_terms(wh, reg_mode, context)
        if not terms:
            return
        print("Objective terms:")
        for key, value in terms.items():
            print(f"{key:>26}: {value:.6e}")

    def _evaluate_terms(self, wh: fem.Function, reg_mode: str, context) -> dict[str, float]:
        return {}


class MinimumResidualSolver(BaseAirflowSolver):

    def _build_system(self, wh_prev: fem.Function, reg_mode: str):
        W = wh_prev.function_space
        domain = W.mesh

        nu = fem.Constant(domain, PETSc.ScalarType(self.ctx.viscosity))
        beta = fem.Constant(domain, PETSc.ScalarType(self.ctx.weight_pde_res))
        gamma = fem.Constant(domain, PETSc.ScalarType(self.ctx.weight_reg))

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

        aF, LF = fem.form(a_pde + a_reg), fem.form(L)
        A = assemble_matrix(aF, bcs=self.ctx.bcs)
        A.assemble()
        b = assemble_vector(LF)
        fem.apply_lifting(b, [aF], bcs=[self.ctx.bcs])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        fem.set_bc(b, self.ctx.bcs)

        S = PETSc.Mat().createAIJ(A.getSizes(), nnz=1, comm=A.comm)
        S.setUp()
        for i in map(int, self.ctx.measurement_ids_W):
            S.setValue(i, i, 1.0)
        S.assemble()

        rhs_add = self.ctx.w_measured.x.petsc_vec.duplicate()
        S.mult(self.ctx.w_measured.x.petsc_vec, rhs_add)

        A.axpy(
            self.ctx.weight_misfit,
            S,
            structure=PETSc.Mat.Structure.DIFFERENT_NONZERO_PATTERN,
        )
        b.axpy(self.ctx.weight_misfit, rhs_add)
        return A, b

    def _solve_step(self, wh_prev: fem.Function, wh: fem.Function, reg_mode: str, context):
        A, b = self._build_system(wh_prev, reg_mode)
        ksp = PETSc.KSP().create(A.comm)
        ksp.setOperators(A)
        ksp.setType("preonly")
        ksp.getPC().setType("lu")
        ksp.setFromOptions()
        ksp.solve(b, wh.x.petsc_vec)
        wh.x.petsc_vec.assemblyBegin()
        wh.x.petsc_vec.assemblyEnd()
        wh.x.array[:] = wh.x.petsc_vec.getArray(readonly=True)

    def _evaluate_terms(self, wh: fem.Function, reg_mode: str, context) -> dict[str, float]:
        domain = self.ctx.domain
        nu = fem.Constant(domain, PETSc.ScalarType(self.ctx.viscosity))

        uh, ph = ufl.split(wh)
        Rmom = -nu * div(grad(uh)) + dot(uh, grad(uh)) + grad(ph)
        Rdiv = div(uh)

        pde_form = fem.form((inner(Rmom, Rmom) + Rdiv * Rdiv) * dx)
        pde = float(domain.comm.allreduce(fem.assemble_scalar(pde_form), op=MPI.SUM))
        reg = self._value_regularization(wh, reg_mode)
        misfit = self._measurement_misfit(wh)

        return {
            "pde_unweighted": pde,
            "reg_unweighted": reg,
            "misfit_unweighted": misfit,
            "pde_weighted": self.ctx.weight_pde_res * pde,
            "reg_weighted": self.ctx.weight_reg * reg,
            "misfit_weighted": self.ctx.weight_misfit * misfit,
            "objective_total_weighted": (
                self.ctx.weight_pde_res * pde
                + self.ctx.weight_reg * reg
                + self.ctx.weight_misfit * misfit
            ),
        }


class WeakPenaltySolver(BaseAirflowSolver):

    def _build_system(self, wh_prev: fem.Function, reg_mode: str):
        uh_prev, _ = wh_prev.split()
        (u, p) = ufl.TrialFunctions(self.ctx.W)
        (v, q) = ufl.TestFunctions(self.ctx.W)
        domain = self.ctx.domain

        nu = fem.Constant(domain, PETSc.ScalarType(self.ctx.viscosity))
        w_pde = fem.Constant(domain, PETSc.ScalarType(self.ctx.weight_pde_res))
        w_reg = fem.Constant(domain, PETSc.ScalarType(self.ctx.weight_reg))

        if reg_mode == "value":
            reg_term = inner(u, v)
        else:
            reg_term = inner(grad(u), grad(v))

        a_form = fem.form((
            w_pde * (
                inner(nu * grad(u), grad(v))
                + inner(grad(u) * uh_prev, v)
                - p * div(v)
                + q * div(u)
            )
            + w_reg * reg_term
        ) * dx)

        zero_vec = fem.Constant(domain, PETSc.ScalarType((0.0,) * domain.geometry.dim))
        L_form = fem.form(inner(zero_vec, v) * dx)

        K_reg = assemble_matrix(a_form)
        K_reg.assemble()
        f_reg = assemble_vector(L_form)

        R = PETSc.Mat().createAIJ(K_reg.getSizes(), nnz=1, comm=K_reg.comm)
        R.setUp()
        r_vec = f_reg.duplicate()
        r_vec.set(0.0)

        for bc in self.ctx.bcs:
            for dof in bc.dof_indices()[0]:
                R.setValue(dof, dof, 1.0)
        R.assemble()

        S = PETSc.Mat().createAIJ(K_reg.getSizes(), nnz=1, comm=K_reg.comm)
        S.setUp()
        for i in map(int, self.ctx.measurement_ids_W):
            S.setValue(i, i, 1.0)
        S.assemble()

        s_vec = self.ctx.w_measured.x.petsc_vec.duplicate()
        S.mult(self.ctx.w_measured.x.petsc_vec, s_vec)

        K_reg.axpy(
            self.ctx.weight_boundary,
            R,
            structure=PETSc.Mat.Structure.DIFFERENT_NONZERO_PATTERN,
        )
        K_reg.axpy(
            self.ctx.weight_misfit,
            S,
            structure=PETSc.Mat.Structure.DIFFERENT_NONZERO_PATTERN,
        )

        f_reg.axpy(self.ctx.weight_boundary, r_vec)
        f_reg.axpy(self.ctx.weight_misfit, s_vec)

        diag = K_reg.getDiagonal()
        if diag.min()[0] < 1e-15:
            print(f"WARNUNG: Matrix hat extrem kleine Diagonalelemente! Min: {diag.min()[0]}")
        if np.any(np.isnan(f_reg.array)):
            print("FEHLER: f_reg enthält NaNs!")

        return K_reg, f_reg

    def _solve_step(self, wh_prev: fem.Function, wh: fem.Function, reg_mode: str, context):
        A, b = self._build_system(wh_prev, reg_mode)
        ksp = PETSc.KSP().create(A.comm)
        ksp.setOperators(A)
        ksp.setType("preonly")
        ksp.getPC().setType("lu")
        ksp.setFromOptions()
        ksp.solve(b, wh.x.petsc_vec)
        wh.x.petsc_vec.assemblyBegin()
        wh.x.petsc_vec.assemblyEnd()
        wh.x.array[:] = wh.x.petsc_vec.getArray(readonly=True)

    def _evaluate_terms(self, wh: fem.Function, reg_mode: str, context) -> dict[str, float]:
        pde = self._weak_form_residual(wh)
        reg = self._value_regularization(wh, reg_mode)
        boundary = self._boundary_penalty(wh)
        misfit = self._measurement_misfit(wh)

        return {
            "pde_unweighted": pde,
            "reg_unweighted": reg,
            "boundary_unweighted": boundary,
            "misfit_unweighted": misfit,
            "pde_weighted": self.ctx.weight_pde_res * pde,
            "reg_weighted": self.ctx.weight_reg * reg,
            "boundary_weighted": self.ctx.weight_boundary * boundary,
            "misfit_weighted": self.ctx.weight_misfit * misfit,
            "objective_total_weighted": (
                self.ctx.weight_pde_res * pde
                + self.ctx.weight_reg * reg
                + self.ctx.weight_boundary * boundary
                + self.ctx.weight_misfit * misfit
            ),
        }


class LinearLeastSquaresSolver(BaseAirflowSolver):

    def __init__(self, ctx: AirflowSolverConfig):
        super().__init__(ctx)
        self._smooth_regularization_operator: sps.csr_matrix | None = None

    def _build_linear_regularization_operator(self, reg_mode: str) -> sps.csr_matrix:
        num_total_dofs = self._num_dofs(self.ctx.W)

        if reg_mode == "value":
            v_map = self.ctx.W.sub(0).dofmap
            u_dofs = np.unique(v_map.list.flatten())
            rows = np.arange(len(u_dofs))
            return sps.csr_matrix(
                (np.ones_like(u_dofs, dtype=float), (rows, u_dofs)),
                shape=(len(u_dofs), num_total_dofs),
            )

        if self._smooth_regularization_operator is not None:
            return self._smooth_regularization_operator

        coords = np.asarray(
            self.ctx.V.tabulate_dof_coordinates(),
            dtype=float,
        )[:, :self.ctx.domain.geometry.dim]
        num_points = len(coords)
        if num_points < 2:
            self._smooth_regularization_operator = sps.csr_matrix((0, num_total_dofs))
            return self._smooth_regularization_operator

        k = min(5, num_points)
        tree = cKDTree(coords)
        dists, neighbors = tree.query(coords, k=k, p=2.0, workers=-1)

        rows, cols, vals = [], [], []
        row_id = 0
        seen_edges: set[tuple[int, int]] = set()
        dim = self.ctx.domain.geometry.dim

        for i in range(num_points):
            for dist_ij, j in zip(np.atleast_1d(dists[i])[1:], np.atleast_1d(neighbors[i])[1:]):
                edge = (i, int(j)) if i < int(j) else (int(j), i)
                if edge[0] == edge[1] or edge in seen_edges:
                    continue
                seen_edges.add(edge)
                weight = 1.0 / max(float(dist_ij), 1e-12)
                for comp in range(dim):
                    wi = int(self.ctx.V_to_W[i * dim + comp])
                    wj = int(self.ctx.V_to_W[int(j) * dim + comp])
                    rows.extend([row_id, row_id])
                    cols.extend([wi, wj])
                    vals.extend([weight, -weight])
                    row_id += 1

        self._smooth_regularization_operator = sps.csr_matrix(
            (vals, (rows, cols)),
            shape=(row_id, num_total_dofs),
        )
        return self._smooth_regularization_operator

    def _prepare_solve(self, reg_mode: str):
        num_total_dofs = self._num_dofs(self.ctx.W)
        reg_op = self._build_linear_regularization_operator(reg_mode)

        bc_rows, bc_cols, bc_vals = [], [], []
        for bc in self.ctx.bcs:
            for dof in bc.dof_indices()[0]:
                bc_rows.append(len(bc_rows))
                bc_cols.append(dof)
                bc_vals.append(1.0)
        R_sp = sps.csr_matrix((bc_vals, (bc_rows, bc_cols)), shape=(len(bc_rows), num_total_dofs))

        m_idx = self.ctx.measurement_ids_W
        M_sp = sps.csr_matrix(
            (np.ones_like(m_idx, dtype=float), (np.arange(len(m_idx)), m_idx)),
            shape=(len(m_idx), num_total_dofs),
        )

        fixed_blocks = [
            np.sqrt(self.ctx.weight_boundary) * R_sp,
            np.sqrt(self.ctx.weight_misfit) * M_sp,
            np.sqrt(self.ctx.weight_reg) * reg_op,
        ]
        fixed_rhs = [
            np.zeros(len(bc_rows)),
            np.sqrt(self.ctx.weight_misfit) * self.ctx.w_measured.x.array[m_idx],
            np.zeros(reg_op.shape[0]),
        ]

        return {
            "num_total_dofs": num_total_dofs,
            "sqrt_w_pde": np.sqrt(self.ctx.weight_pde_res),
            "reg_op": reg_op,
            "fixed_matrix": sps.vstack(fixed_blocks).tocsr(),
            "fixed_rhs": np.concatenate(fixed_rhs),
        }

    def _solve_step(self, wh_prev: fem.Function, wh: fem.Function, reg_mode: str, context):
        K_petsc, f_petsc = self._build_weak_form_system(wh_prev)
        ai, aj, av = K_petsc.getValuesCSR()
        K_sp = sps.csr_matrix((av, aj, ai), shape=(context["num_total_dofs"], context["num_total_dofs"]))

        A_stack = sps.vstack([
            context["sqrt_w_pde"] * K_sp,
            context["fixed_matrix"],
        ]).tocsr()
        b_stack = np.concatenate([
            -context["sqrt_w_pde"] * f_petsc.array,
            context["fixed_rhs"],
        ])

        wh.x.array[:] = lsqr(A_stack, b_stack, iter_lim=5000)[0]

    def _evaluate_terms(self, wh: fem.Function, reg_mode: str, context) -> dict[str, float]:
        K_petsc, f_petsc = self._build_weak_form_system(wh)
        residual = f_petsc.duplicate()
        K_petsc.mult(wh.x.petsc_vec, residual)
        residual.axpy(-1.0, f_petsc)
        pde = float(residual.norm(PETSc.NormType.NORM_2) ** 2)

        boundary = self._boundary_penalty(wh)
        misfit = self._measurement_misfit(wh)
        reg_vec = context["reg_op"] @ wh.x.array
        reg = float(np.dot(reg_vec, reg_vec))

        return {
            "pde_unweighted": pde,
            "reg_unweighted": reg,
            "boundary_unweighted": boundary,
            "misfit_unweighted": misfit,
            "pde_weighted": self.ctx.weight_pde_res * pde,
            "reg_weighted": self.ctx.weight_reg * reg,
            "boundary_weighted": self.ctx.weight_boundary * boundary,
            "misfit_weighted": self.ctx.weight_misfit * misfit,
            "objective_total_weighted": (
                self.ctx.weight_pde_res * pde
                + self.ctx.weight_reg * reg
                + self.ctx.weight_boundary * boundary
                + self.ctx.weight_misfit * misfit
            ),
        }
