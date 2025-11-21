import numpy as np
import ufl
from ufl import inner, grad, dx, sqrt, dot, CellDiameter

from dolfinx import fem, mesh
from basix.ufl import element
from dolfinx.fem.petsc import LinearProblem
from petsc4py import PETSc


class GasSourceEstimator:

    def __init__(self, 
                domain: mesh.Mesh, 
                wind: fem.Function | None = None, 
                D_phys: float = 1e-3):

        self.domain = domain

        # ----------------------------------------------------
        # Wind field
        # ----------------------------------------------------
        if wind is None:
            Vwind = fem.functionspace(
                domain,
                element("CG", domain.basix_cell(), 2, shape=(domain.geometry.dim,))
            )
            wind = fem.Function(Vwind)
            wind.x.array[:] = 0.0
            self.has_wind = False
        else:
            self.has_wind = True

        self.wind_field = wind

        self.D_phys = fem.Constant(domain, PETSc.ScalarType(D_phys))

        w_elem = wind.function_space.ufl_element()
        if hasattr(w_elem, "sub_elements"):
            degree = w_elem.sub_elements[0].degree
        else:
            degree = w_elem.degree

        if isinstance(degree, tuple):
            degree = max(degree)

        self.scalar_space = fem.functionspace(
            domain, element("CG", domain.basix_cell(), degree)
        )

        self.source_est = fem.Function(self.scalar_space, name="source")
        self.residual   = fem.Function(self.scalar_space, name="residual")
        self.c_est      = None
        self.f_true     = None

        coords = domain.geometry.x
        self.x_min, self.y_min = np.min(coords[:, :2], axis=0)
        self.x_max, self.y_max = np.max(coords[:, :2], axis=0)
        self.Lx = self.x_max - self.x_min
        self.Ly = self.y_max - self.y_min

        self.tau = self._build_supg_parameters()
        self.bcs: list[fem.DirichletBC] = [self._build_default_bc()]

        self.forward_problem = self._build_forward_problem()
        self.adjoint_problem = self._build_adjoint_problem()

        self.m_ids = None
        self.m     = None

    def set_true_gaussian_source(self, x: float, y: float, sigma: float, amplitude: float = 1.0):
        """
        Convenience function to define a Gaussian true source centered at (x, y).
        """
        f_new = fem.Function(self.scalar_space, name="f_true_gaussian")

        def gaussian(xx):
            x0, y0 = x, y
            return amplitude * np.exp(
                -((xx[0] - x0) ** 2 + (xx[1] - y0) ** 2) / (2 * sigma**2)
            )

        f_new.interpolate(gaussian)

        # Store as ground truth (also updates ground-truth concentration)
        self.set_ground_truth(f_new)

        return f_new

    def set_ground_truth(self, f_true: fem.Function):
        if f_true.function_space != self.scalar_space:
            raise ValueError("Ground truth f_true must be in the same function space as estimator.")
        
        self.f_true = fem.Function(self.scalar_space, name="f_true")
        self.f_true.x.array[:] = f_true.x.array.copy()
    
    def get_ground_truth_concentration(self):
        if self.f_true is None:
            raise RuntimeError("No ground truth source has been set.")

        # Backup 
        backup = self.source_est.x.array.copy()

        # Use true source
        self.source_est.x.array[:] = self.f_true.x.array
        forward_problem = self._build_forward_problem()
        c_true = forward_problem.solve()

        # Restore actual inversion
        self.source_est.x.array[:] = backup

        return c_true
    
    def reset_random_measurements(self, p: int, seed: int = 1):
        if self.f_true is None:
            raise RuntimeError("Set ground truth f_true first using set_ground_truth().")

        # Always recompute GT concentration cleanly
        c_true = self.get_ground_truth_concentration()

        rng = np.random.default_rng(seed)
        n = c_true.x.array.size

        m_ids = rng.choice(np.arange(n), size=p, replace=False)
        m_values = c_true.x.array[m_ids].copy()

        self.set_measurements(m_ids, m_values)

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
    
    def get_measurement_coordinates(self) -> np.ndarray:
        if self.m_ids is None:
            raise RuntimeError("No gas measurements set yet.")
        
        coords = self.scalar_space.tabulate_dof_coordinates()
        return coords[self.m_ids]

    def solve_forward(self) -> fem.Function:
        if not self.has_wind:
            raise RuntimeError("No wind field set yet, but is required to solve forward problem.")
        self.c = self.forward_problem.solve()
        return self.c
    
    def reset_wind(self, wind: fem.Function):
        """
        Update wind field and all members/structures depending on it to allow reusing the instance
        for multiple experiments without creating a new GasSourceEstimator every time.
        """
        # reset
        self.wind_field = wind
        self.has_wind = True
        self.tau = self._build_supg_parameters()

        self.forward_problem = self._build_forward_problem()
        self.adjoint_problem = self._build_adjoint_problem()

        # Depend on wind field as well
        self.m_ids = None
        self.m = None

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
        f = self.source_est
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
                f.x.array[:] = np.maximum(f.x.array, 0.0)

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
        self.c_est = self.solve_forward()

        return self.source_est

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
        f = self.source_est
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
                f.x.array[:] = np.maximum(f.x.array, 0.0)
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
        self.c_est = self.solve_forward()

        return self.source_est

    def estimated_source_max_location(self) -> np.ndarray:
        """
        Liefert die Koordinate des DOFs, an dem f(x) maximal ist.
        """
        coords = self.scalar_space.tabulate_dof_coordinates()
        idx_max = np.argmax(self.source_est.x.array)
        return coords[idx_max]

    def estimated_source_max_value(self) -> float:
        return float(np.max(self.source_est.x.array))
    
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
        u = ufl.TrialFunction(self.scalar_space)
        v = ufl.TestFunction(self.scalar_space)
        beta = self.wind_field
        f = self.source_est
        D = self.D_phys
        tau = self.tau

        a = (D * inner(grad(u), grad(v)) * dx
             + inner(beta, grad(u)) * v * dx
             + tau * inner(beta, grad(u)) * inner(beta, grad(v)) * dx)

        L = f * v * dx + tau * f * inner(beta, grad(v)) * dx

        return LinearProblem(a, L, self.bcs)

    def _build_adjoint_problem(self):
        v = ufl.TestFunction(self.scalar_space)
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