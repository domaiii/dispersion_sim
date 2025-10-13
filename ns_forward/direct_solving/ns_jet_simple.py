from dolfinx import fem, mesh
from dolfinx.fem.petsc import NonlinearProblem
from mpi4py import MPI
import numpy as np
from petsc4py.PETSc import ScalarType, KSP
from petsc4py import PETSc

from ns_helpers import *

x_lim = 20.0
y_lim = 5.0
x_res = 128
y_res = 64

domain = create_2d_mesh_rect(x_lim, y_lim, x_res, y_res, mesh.CellType.quadrilateral)
V = create_2d_th_functionspace(domain)

w = fem.Function(V)
w.x.array[:] = 0.0

#nu_target = 14.9e-6 # Not working. Requires refined net close to the inlet
nu_target = 14.9e-5
nu_start = 1.0     
num_continuation_steps = 10

nu_fem_constant = fem.Constant(domain, ScalarType(nu_start)) 

F, J = define_ns_ufl_forms(w, V, nu_fem_constant)

def no_slip_bdry(x: np.ndarray) -> np.ndarray:
    return np.logical_or(np.isclose(x[1], 0), np.isclose(x[1], y_lim))

def inflow_entire_left_boundary(x: np.ndarray) -> np.ndarray:
    return np.isclose(x[0], 0)

def outflow(x: np.ndarray) -> np.ndarray:
    return np.isclose(x[0], x_lim)

# def inflow_velocity_profile_nozzle(x: np.ndarray) -> np.ndarray:
#     is_in_inflow_range = np.logical_and(x[1] >= 0.375 * y_lim, x[1] <= 0.625 * y_lim)
#     ux_val = np.where(is_in_inflow_range, 1.0, 0.0)
#     return np.stack((ux_val, np.zeros_like(x[1])))

def inflow_profile_parabolic_middle(x: np.ndarray) -> np.ndarray:
    y_start_nozzle = 0.48 * y_lim
    y_end_nozzle = 0.52 * y_lim
    nozzle_height = y_end_nozzle - y_start_nozzle 

    y_coords = x[1]
    
    ux_val = np.zeros_like(y_coords) 

    in_nozzle = np.logical_and(y_coords >= y_start_nozzle, y_coords <= y_end_nozzle)
    y_relative = y_coords[in_nozzle] - y_start_nozzle 
    
    U_max = 0.2
    
    ux_val[in_nozzle] = U_max * 4.0 * y_relative * (nozzle_height - y_relative) / nozzle_height**2
    
    return np.stack((ux_val, np.zeros_like(y_coords)))

bc_no_slip = create_noslip_dirichlet_bc(V, no_slip_bdry)
bc_in = velocity_profile_bc(V, inflow_entire_left_boundary, inflow_profile_parabolic_middle)
# bc_out = constant_pressure_bc(V, outflow)

bcs = [bc_in, bc_no_slip]

problem = NonlinearProblem(F, w, bcs, J)
solver = setup_newton_solver(problem, max_iterations=100, 
                             ksp_type=KSP.Type.GMRES, 
                             pc_type=PETSc.PC.Type.LU, 
                             tolerance=1e-7)
setup_pressure_nullspace(V, solver.krylov_solver)

all_converged, _ = solve_with_continuation(problem, solver, w, nu_fem_constant, 
                                           nu_target, num_continuation_steps, nu_start)

if all_converged:
    u_vis, p_vis = save_solution_for_visualization(w, domain, output_dir_path="./output_results")
else:
    if MPI.COMM_WORLD.rank == 0:
        print("Solution not saved as solver procedure did not complete successfully.")

