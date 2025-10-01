from dolfinx import fem, mesh
from dolfinx.fem.petsc import NonlinearProblem
from mpi4py import MPI
import numpy as np
import sys
from petsc4py.PETSc import ScalarType, KSP
from petsc4py import PETSc

from ns_helpers import *

if __name__ == "__main__":
    x_lim = 2.0
    y_lim = 1.0
    x_res = 200
    y_res = 100
    
    domain = create_2d_mesh_rect(x_lim, y_lim, x_res, y_res, mesh.CellType.quadrilateral)
    V = create_2d_th_functionspace(domain)
    
    w = fem.Function(V)
    w.x.array[:] = 0.0
    nu = fem.Constant(domain, ScalarType(14.9e-6))  # m²/s

    F, J = define_ns_ufl_forms(w, V, nu, False)
    
    def no_slip_bdry(x: np.ndarray) -> np.ndarray:
        top_and_right =  np.logical_or(np.isclose(x[0], x_lim), np.isclose(x[1], y_lim))
        lower_left = np.logical_and(np.isclose(x[0], 0.0), x[1] < y_lim / 2.0)

        return np.logical_or(top_and_right, lower_left)
    
    def inflow(x: np.ndarray) -> np.ndarray:
        return np.logical_and(np.isclose(x[0], 0.0), x[1] > y_lim / 2.0)

    def outflow(x: np.ndarray) -> np.ndarray:
        return np.isclose(x[1], 0.0)

    def inflow_velocity_profile(x: np.ndarray) -> np.ndarray:
        y_centered = (x[1] - y_lim/2) / (y_lim/2)
        u_val = np.where(
            inflow(x),
            4.0 * (1 - y_centered) * y_centered,
            0.0
        )
        return np.stack((u_val, np.zeros_like(x[1])))
    
    # def inflow_velocity_profile(x: np.ndarray) -> np.ndarray:
    #     y_lim = 1.0
    #     is_in_inflow_range = np.logical_and(x[1] >= 0.375 * y_lim, x[1] <= 0.625 * y_lim)
    #     ux_val = np.where(is_in_inflow_range, 1.0, 0.0)

        # return np.stack((ux_val, np.zeros_like(x[1])))
    
    bc_no_slip = create_noslip_dirichlet_bc(V, no_slip_bdry)
    bc_in = velocity_profile_bc(V, inflow, inflow_velocity_profile)
    bc_out = constant_pressure_bc(V, outflow)
    
    bcs = [bc_in, bc_no_slip, bc_out]

    problem = NonlinearProblem(F, w, bcs, J)
    solver = setup_newton_solver(problem, max_iterations=50, ksp_type=KSP.Type.GMRES, 
                                 pc_type=PETSc.PC.Type.HYPRE)
    
    solve_adaptive_continuation(problem, solver, w, nu, nu_target=1.0e-4, nu_start=1.0)

    u_vis, p_vis = save_solution_for_visualization(w, domain, output_dir_path="./output_results")