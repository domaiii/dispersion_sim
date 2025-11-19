import numpy as np
import ufl
from ufl import inner, grad, dx, sqrt, dot, CellDiameter

from dolfinx import fem, mesh
from basix.ufl import element
from dolfinx.fem.petsc import LinearProblem
from petsc4py import PETSc


class GasSourceEstimator:

    def __init__(self, domain: mesh.Mesh, wind: fem.Function, D_phys: float | None = 1e-3):
        self.domain = domain
        self.wind_field = wind
        if D_phys is None:
            D_phys = 1e-3
        self.D_phys = fem.Constant(domain, PETSc.ScalarType(D_phys))

        # Derive function space
        elem = wind.function_space.ufl_element()
        degree = elem.sub_elements[0].degree if hasattr(elem, "sub_elements") else elem.degree
        if isinstance(degree, tuple):
            degree = max(degree)

        self.scalar_space = fem.functionspace(domain, element("CG", domain.basix_cell(), degree))

        self.u = ufl.TrialFunction(self.scalar_space)
        self.v = ufl.TestFunction(self.scalar_space)

        self.source = fem.Function(self.scalar_space, name="source")
        self.concentration = fem.Function(self.scalar_space, name="concentration")
        self.residual = fem.Function(self.scalar_space, name="residual")

        # Geometry
        coords = domain.geometry.x
        self.x_min, self.y_min = np.min(coords[:, 0]), np.min(coords[:, 1])
        self.x_max, self.y_max = np.max(coords[:, 0]), np.max(coords[:, 1])
        self.Lx = self.x_max - self.x_min
        self.Ly = self.y_max - self.y_min

        # SUPG parameter
        self.tau = self._build_supg_parameters()

        # BCs
        self.bcs: list[fem.DirichletBC] = []
        self.bcs.append(self._build_default_bc())

        # Build problems
        self.forward_problem = self._build_forward_problem()
        self.adjoint_problem = self._build_adjoint_problem()

        # Measurements
        self.m_ids = None
        self.m = None

        # Optional ground truth
        self.f_true = None
        self.c_true = None

    def set_ground_truth(self, f_true: fem.Function):
        if f_true.function_space != self.scalar_space:
            raise ValueError("Ground truth f_true must be in the same function space as estimator.")
        
        self.f_true = fem.Function(self.scalar_space, name="f_true")
        self.f_true.x.array[:] = f_true.x.array.copy()
        self.compute_ground_truth_concentration()

        return self.f_true
    
    def compute_ground_truth_concentration(self):
        """Compute gas source distribution based on ground truth source and given wind field."""
        if self.f_true is None:
            raise RuntimeError("No ground truth source. Use set_ground_truth(...). first")
        
        backup = self.source.x.array.copy()
        self.source.x.array[:] = self.f_true.x.array
        
        self.c_true = self.solve_forward()
        
        self.source.x.array[:] = backup
        
        return self.c_true
    
    def generate_measurements_from_ground_truth(self, p: int, seed: int = 1):
        if self.c_true is None:
            raise RuntimeError("Compute ground truth concentration with " \
            "compute_ground_truth_concentration() first.")
        
        rng = np.random.default_rng(seed)
        n = self.c_true.x.array.size
        
        m_ids = rng.choice(np.arange(n), size=p, replace=False)
        m_values = self.c_true.x.array[m_ids].copy()
        self.set_measurements(m_ids, m_values)
        
        return m_ids, m_values

    def _build_default_bc(self):
        """
        Default: c=0 at x = x_min.
        """
        def boundary(x):
            return np.isclose(x[0], self.x_min)

        u_zero = fem.Constant(self.domain, PETSc.ScalarType(0.0))
        Q = self.scalar_space
        bc_zero = fem.dirichletbc(u_zero, fem.locate_dofs_geometrical(Q, boundary), Q)
        
        return bc_zero
    
    def reset_dirichlet_bcs(self, bc: fem.DirichletBC | list[fem.DirichletBC]):
        self.bcs = []
        self.add_dirichlet_bc(bc)

    def add_dirichlet_bc(self, bc: fem.DirichletBC | list[fem.DirichletBC]):
        if isinstance(bc, list):
            self.bcs += bc
        else:
            self.bcs.append(bc)

        self.forward_problem = self._build_forward_problem()
        self.adjoint_problem = self._build_adjoint_problem()

    def set_measurements(self, m_ids: np.ndarray, m_values: np.ndarray):
        self.m_ids = np.asarray(m_ids, dtype=np.int32)
        self.m = np.asarray(m_values, dtype=float)

    def set_measurements_from_distribution(self, c_true: fem.Function, m_ids: np.ndarray):
        self.m_ids = np.asarray(m_ids, dtype=np.int32)
        self.m = c_true.x.array[self.m_ids].copy()

    def solve_forward(self) -> fem.Function:
        self.c = self.forward_problem.solve()
        return self.c
    
    def reinitialize(self, wind: fem.Function):
        """
        Update wind field and all members/structures depending on it to allow reusing the instance
        for multiple experiments without creating a new GasSourceEstimator every time.
        """
        # reset
        self.wind_field = wind
        self.tau = self._build_supg_parameters()

        self.forward_problem = self._build_forward_problem()
        self.adjoint_problem = self._build_adjoint_problem()

        # Depend on wind field as well
        self.m_ids = None
        self.m = None
        self.c_true = None

    def solve_L1(self,
                 gamma_reg: float = 1e-2,
                 max_it: int = 50,
                 alpha0: float = 1.0,
                 tol_rel: float = 1e-4,
                 tol_grad: float = 1e-4,
                 c_arm: float = 1e-3,
                 rho: float = 0.5,
                 verbose: bool = True):
        
        if self.m_ids is None or self.m is None:
            raise RuntimeError("No measurements set. Call set_measurements(...) first.")

        m_ids = self.m_ids
        m = self.m
        f = self.source
        residual = self.residual

        f.x.array[:] = 0.0  # Start at f=0

        misfit_hist = []
        alpha_hist = []
        alpha = alpha0 / rho

        def compute_misfit(c_func: fem.Function, f_func: fem.Function):
            r = c_func.x.array[m_ids] - m
            return np.dot(r, r) + 0.5 * gamma_reg * np.sum(np.abs(f_func.x.array))

        for it in range(max_it):

            alpha /= rho

            c = self.solve_forward()
            mis = compute_misfit(c, f)

            residual.x.array[:] = 0.0
            residual.x.array[m_ids] = -(c.x.array[m_ids] - m)
            adj = self.adjoint_problem.solve()

            gradJ = -adj.x.array
            grad_norm = np.linalg.norm(gradJ)

            f_old = f.x.array.copy()

            # Armijo
            alpha_local = alpha
            while True:
                y = f_old - alpha_local * gradJ
                f.x.array[:] = np.sign(y) * np.maximum(np.abs(y) - alpha_local * gamma_reg, 0.0)

                c_trial = self.solve_forward()
                mis_trial = compute_misfit(c_trial, f)

                if mis_trial <= mis - c_arm * alpha_local * grad_norm**2:
                    break

                alpha_local *= rho
                if alpha_local < 1e-10:
                    print("Step size too small.")
                    break

            alpha = alpha_local

            misfit_hist.append(mis_trial)
            alpha_hist.append(alpha)

            if verbose and (it % 10 == 0 or it == max_it - 1):
                print(f"it {it:3d} mis={mis:.3e} ||grad||={grad_norm:.3e} α={alpha:.2e}")

            # Convergence criteria
            if it > 1:
                rel_change = abs(misfit_hist[-2] - misfit_hist[-1]) / max(1e-12, misfit_hist[-2])
                if rel_change < tol_rel or grad_norm < tol_grad:
                    if verbose:
                        print(f"Stopped at it {it}, rel_change={rel_change:.3e}")
                    break

        self.misfit_hist_L1 = np.array(misfit_hist)
        self.alpha_hist_L1 = np.array(alpha_hist)
        return self.source

    def solve_L2(self,
                gamma_reg: float = 1e-2,
                max_it: int = 200,
                alpha0: float = 1.0,
                tol_rel: float = 1e-7,
                tol_grad: float = 1e-8,
                c_arm: float = 1e-3,
                rho: float = 0.5,
                verbose: bool = True):

        if self.m_ids is None or self.m is None:
            raise RuntimeError("No measurements set. Use set_measurements(...) first.")

        m_ids = self.m_ids
        m = self.m
        f = self.source
        residual = self.residual

        f.x.array[:] = 0.0

        misfit_hist = []
        alpha_hist = []

        # Schrittweite starten
        alpha = alpha0 / rho

        def compute_misfit(c_func: fem.Function, f_func: fem.Function):
            r = c_func.x.array[m_ids] - m
            return 0.5 * np.dot(r, r) + 0.5 * gamma_reg * np.dot(f_func.x.array, f_func.x.array)

        for it in range(max_it):

            # Schrittweite für neue Iteration erhöhen
            alpha /= rho

            c = self.solve_forward()
            mis = compute_misfit(c, f)

            # Adjoint RHS
            residual.x.array[:] = 0.0
            residual.x.array[m_ids] = -(c.x.array[m_ids] - m)
            adj = self.adjoint_problem.solve()

            # Gradient
            gradJ = -adj.x.array + gamma_reg * f.x.array
            grad_norm = np.linalg.norm(gradJ)

            f_old = f.x.array.copy()

            # Local Armijo
            alpha_local = alpha
            while True:
                f.x.array[:] = f_old - alpha_local * gradJ

                c_trial = self.solve_forward()
                mis_trial = compute_misfit(c_trial, f)

                if mis_trial <= mis - c_arm * alpha_local * grad_norm**2:
                    break

                alpha_local *= rho
                if alpha_local < 1e-14:
                    print("Step size too small.")
                    break

            # Übernommene Schrittweite für nächste Iteration merken
            alpha = alpha_local

            misfit_hist.append(mis_trial)
            alpha_hist.append(alpha)

            if verbose and (it % 10 == 0 or it == max_it - 1):
                print(f"it {it:3d} mis={mis:.3e} ||grad||={grad_norm:.3e} α={alpha:.2e}")

            # Convergence
            if it > 1:
                rel_change = abs(misfit_hist[-2] - misfit_hist[-1]) / max(1e-12, misfit_hist[-2])
                if rel_change < tol_rel or grad_norm < tol_grad:
                    if verbose:
                        print(f"Stopped at it {it}, rel_change={rel_change:.3e}")
                    break

        self.misfit_hist_L2 = np.array(misfit_hist)
        self.alpha_hist_L2 = np.array(alpha_hist)
        return self.source

    def estimated_source_max_location(self) -> np.ndarray:
        """
        Liefert die Koordinate des DOFs, an dem f(x) maximal ist.
        """
        coords = self.scalar_space.tabulate_dof_coordinates()
        idx_max = np.argmax(self.source.x.array)
        return coords[idx_max]

    def estimated_source_max_value(self) -> float:
        return float(np.max(self.source.x.array))
    
    def _build_supg_parameters(self):
        beta = self.wind_field
        h = CellDiameter(self.domain)

        U_char = sqrt(dot(beta, beta))
        L_char = fem.Constant(self.domain, PETSc.ScalarType(self.Lx))
        Pe = U_char * L_char / self.D_phys
        nb = sqrt(inner(beta, beta))

        tau = 0.5 * h * pow(4.0 / (Pe * h) + 2.0 * nb, -1.0)
        return tau

    def _build_forward_problem(self):
        u, v = self.u, self.v
        beta = self.wind_field
        f = self.source
        D = self.D_phys
        tau = self.tau

        a = (D * inner(grad(u), grad(v)) * dx
             + inner(beta, grad(u)) * v * dx
             + tau * inner(beta, grad(u)) * inner(beta, grad(v)) * dx)

        L = f * v * dx + tau * f * inner(beta, grad(v)) * dx

        return LinearProblem(a, L, self.bcs)

    def _build_adjoint_problem(self):
        v = self.v
        beta = self.wind_field
        D = self.D_phys
        residual = self.residual
        tau = self.tau

        lam = ufl.TrialFunction(self.scalar_space)

        a_adj = (D * inner(grad(v), grad(lam)) * dx
                 + inner(beta, grad(v)) * lam * dx
                 + tau * inner(beta, grad(v)) * inner(beta, grad(lam)) * dx)

        L_adj = (inner(residual, v) * dx
                 + tau * inner(beta, grad(v)) * residual * dx)

        return LinearProblem(a_adj, L_adj, self.bcs)