from pathlib import Path
from dolfinx.io import XDMFFile
from typing import Tuple, Callable, Optional
from dolfinx import fem, mesh
from dolfinx.mesh import Mesh
from dolfinx.fem.bcs import DirichletBC
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
from dolfinx.fem import Function, FunctionSpace
from mpi4py import MPI
import numpy as np
import ufl
import sys
from basix.ufl import element, mixed_element
from petsc4py.PETSc import ScalarType, KSP
from petsc4py import PETSc
from ufl.form import Form # Explicitly import Form for type hints

# Set PETSc error handler to default for more detailed output in case of errors
PETSc.Sys.popErrorHandler()

def create_2d_mesh_rect(x_lim: float, y_lim: float,
                        x_res: int, y_res: int,
                        cell_type: mesh.CellType) -> Mesh:
    """
    Create a 2D rectangular mesh.

    This function generates a quadrilateral mesh for a rectangular domain
    starting from the origin (0, 0).

    Args:
        x_lim: The extent of the domain in the x-direction.
        y_lim: The extent of the domain in the y-direction.
        x_res: The number of cells in the x-direction.
        y_res: The number of cells in the y-direction.
        cell_type: The type of cell to use for the mesh (e.g., mesh.CellType.quadrilateral).

    Returns:
        A DolfinX Mesh object representing the rectangular domain.
    """
    pts = ((0.0, 0.0), (x_lim, y_lim))
    return mesh.create_rectangle(MPI.COMM_WORLD, pts, (x_res, y_res), cell_type)

def create_2d_th_functionspace(domain: Mesh) -> FunctionSpace:
    """
    Create a Taylor-Hood (P2-P1) mixed function space on a 2D mesh, using 
    Lagrange elements of degree 2 for velocity and degree 1 for pressure.

    Args:
        domain: The DolfinX Mesh object representing the domain.

    Returns:
        A DolfinX FunctionSpace object for the mixed P2-P1 elements.
    """
    elem_u = element("Lagrange", domain.basix_cell(), 2, shape=(domain.topology.dim,))
    elem_p = element("Lagrange", domain.basix_cell(), 1)
    
    mixed_elem = mixed_element([elem_u, elem_p])

    return fem.functionspace(domain, mixed_elem)

def define_ns_ufl_forms(w: Function, V: FunctionSpace, nu: fem.Constant, enable_supg: bool = False) -> Tuple[Form, Form]:
    """
    Defines the UFL forms (residual F and Jacobian J) for the steady-state
    Navier-Stokes equations, with optional SUPG stabilization.

    Args:
        w: The fem.Function representing the current solution vector (u, p).
        V: The mixed function space.
        nu: The kinematic viscosity constant.
        enable_supg: Boolean flag to enable/disable SUPG stabilization.

    Returns:
        A tuple containing the residual form (F) and the Jacobian form (J).
    """
    v, q = ufl.TestFunctions(V)
    u, p = ufl.split(w)
    
    F = (ufl.inner(ufl.dot(u, ufl.nabla_grad(u)), v) * ufl.dx
         + ufl.inner(nu * ufl.grad(u), ufl.grad(v)) * ufl.dx
         - ufl.inner(p, ufl.div(v)) * ufl.dx
         - ufl.inner(ufl.div(u), q) * ufl.dx)

    if enable_supg:
        h = ufl.CellDiameter(V.mesh) 
        u_norm = ufl.sqrt(ufl.dot(u, u) + 1e-12) 

        tau = 1.0 / ufl.sqrt( (2.0 * u_norm / h)**2 + (4.0 * nu / h**2)**2 )
        r_momentum_strong = ufl.dot(u, ufl.nabla_grad(u)) + ufl.grad(p) - nu * ufl.div(ufl.grad(u))
        r_continuity_strong = ufl.div(u)

        F_supg_momentum = ufl.inner(r_momentum_strong, tau * ufl.dot(u, ufl.nabla_grad(v))) * ufl.dx
        F_pspg_continuity = ufl.inner(ufl.grad(q), tau * ufl.grad(r_continuity_strong)) * ufl.dx
        F += F_supg_momentum + F_pspg_continuity

    # Define the Jacobian matrix (J), the derivative of F with respect to 'w'.
    J = ufl.derivative(F, w, ufl.TrialFunction(V))

    return F, J

def create_noslip_dirichlet_bc(V_mixed: FunctionSpace, 
                               boundary_marker: Callable[[np.ndarray], np.ndarray]) -> DirichletBC:
    """
    Creates a Dirichlet boundary condition for a no-slip velocity boundary. It is designed for use
    with a mixed Taylor-Hood (P2-P1) function space.

    Args:
        V_mixed: The mixed function space (e.g., Taylor-Hood P2-P1) where
                 the velocity component is at index 0.
        boundary_marker: A Python function that takes coordinates (x) and
                         returns a boolean array, marking the boundary's location.

    Returns:
        A DirichletBC object for the velocity, setting it to zero on the marked boundary.
    """
    V_velocity_sub = V_mixed.sub(0)
    V_velocity_collapsed, _ = V_velocity_sub.collapse()

    zero_velocity_func = fem.Function(V_velocity_collapsed)
    zero_velocity_func.x.array[:] = 0.0
    
    boundary_dofs = fem.locate_dofs_geometrical((V_velocity_sub, V_velocity_collapsed), boundary_marker)
    
    return fem.dirichletbc(zero_velocity_func, boundary_dofs, V_velocity_sub)

def velocity_profile_bc(
    V_mixed: FunctionSpace,
    marker: Callable[[np.ndarray], np.ndarray],
    velocity_profile: Callable[[np.ndarray], np.ndarray],
    component_index: int = 0,
) -> DirichletBC:
    """
    Creates a Dirichlet boundary condition for a velocity profile.

    Args:
        V_mixed: The mixed function space.
        marker: A function that marks the affected boundary.
        velocity_profile: A function that defines the velocity profile at the affected boundary.
        component_index: The subspace index for velocity (defaults to 0).

    Returns:
        A DirichletBC object for the inflow.
    """
    V_velocity_sub = V_mixed.sub(component_index)
    V_velocity_collapsed, _ = V_velocity_sub.collapse()
    inflow_func = fem.Function(V_velocity_collapsed)
    inflow_func.interpolate(velocity_profile)
    inlet_dofs = fem.locate_dofs_geometrical((V_velocity_sub, V_velocity_collapsed), marker)

    return fem.dirichletbc(inflow_func, inlet_dofs, V_velocity_sub)

def constant_pressure_bc(
    V_mixed: FunctionSpace,
    outflow_marker: Callable[[np.ndarray], np.ndarray],
    pressure_value: ScalarType = ScalarType(0.0),
    component_index: int = 1,  # Pressure is usually the second component
) -> DirichletBC:
    """
    Creates a Dirichlet boundary condition for an outflow.

    Args:
        V_mixed: The mixed function space.
        outflow_marker: A function that marks the outflow boundary.
        pressure_value: The pressure value to set at the outflow (default: 0.0).
        component_index: The subspace index for pressure (defaults to 1).

    Returns:
        A DirichletBC object for the outflow.
    """
    V_pressure_sub = V_mixed.sub(component_index)
    V_pressure_collapsed, _ = V_pressure_sub.collapse()
    pressure_func = fem.Function(V_pressure_collapsed)
    pressure_func.x.array[:] = pressure_value
    outlet_dofs = fem.locate_dofs_geometrical((V_pressure_sub, V_pressure_collapsed), outflow_marker)
    
    return fem.dirichletbc(pressure_func, outlet_dofs, V_pressure_sub)

def pressure_profile_bc(
    V_mixed: FunctionSpace,
    marker: Callable[[np.ndarray], np.ndarray],
    pressure_profile: Callable[[np.ndarray], np.ndarray],
    component_index: int = 1,  # Pressure is usually the second component
) -> DirichletBC:
    """
    Creates a Dirichlet boundary condition for a given pressure profile.

    Args:
        V_mixed: The mixed function space.
        marker: A function that marks the affected boundary.
        pressure_profile: A function that defines the pressure profile.
        component_index: The subspace index for pressure (defaults to 1).

    Returns:
        A DirichletBC object for the outflow.
    """
    V_pressure_sub = V_mixed.sub(component_index)
    V_pressure_collapsed, _ = V_pressure_sub.collapse()
    pressure_func = fem.Function(V_pressure_collapsed)
    pressure_func.interpolate(pressure_profile)
    outlet_dofs = fem.locate_dofs_geometrical((V_pressure_sub, V_pressure_collapsed), marker)
    
    return fem.dirichletbc(pressure_func, outlet_dofs, V_pressure_sub)


def configure_ksp_solver(ksp: PETSc.KSP,
                         ksp_type: Optional[PETSc.KSP.Type] = PETSc.KSP.Type.GMRES,
                         pc_type: Optional[PETSc.PC.Type] = PETSc.PC.Type.LU,
                         rtol: float = 1e-9, atol: float = 1e-10):
    """
    Configures the internal linear solver (KSP) for a nonlinear solver.

    Args:
        ksp: The PETSc KSP object to configure.
        ksp_type: The Krylov subspace method type (e.g., GMRES).
        pc_type: The preconditioner type (e.g., LU).
        rtol: The relative tolerance for the linear solver.
        atol: The absolute tolerance for the linear solver.
    """
    ksp.setType(ksp_type)
    ksp.getPC().setType(pc_type)
    ksp.setTolerances(rtol=rtol, atol=atol)
    ksp.setFromOptions()

def setup_newton_solver(problem: NonlinearProblem, 
                        max_iterations: int = 50, 
                        tolerance: float = 1e-8,
                        comm: MPI.Comm = MPI.COMM_WORLD,
                        ksp_type: Optional[PETSc.KSP.Type] = PETSc.KSP.Type.GMRES,
                        pc_type: Optional[PETSc.PC.Type] = PETSc.PC.Type.LU,
                        ksp_rtol: float = 1e-8,
                        ksp_atol: float = 1e-9) -> NewtonSolver:
    """
    Creates and configures a NewtonSolver for the given problem with detailed
    configuration for the linear solver.

    Args:
        problem: The NonlinearProblem object.
        max_iterations: The maximum number of Newton iterations.
        tolerance: The solver's relative and absolute tolerance for the nonlinear iteration.
        comm: The MPI.Comm channel.
        ksp_type: The Krylov subspace method type (e.g., GMRES).
        pc_type: The preconditioner type (e.g., LU).
        ksp_rtol: The relative tolerance for the linear solver (KSP).
        ksp_atol: The absolute tolerance for the linear solver (KSP).

    Returns:
        A configured NewtonSolver.
    """
    # Use the provided communicator (comm) which defaults to MPI.COMM_WORLD.
    # The 'problem' object does not have a 'comm' attribute.
    solver = NewtonSolver(comm, problem)
    
    ksp = solver.krylov_solver
    configure_ksp_solver(ksp, ksp_type=ksp_type, pc_type=pc_type, rtol=ksp_rtol, atol=ksp_atol)

    solver.atol = tolerance
    solver.rtol = tolerance
    solver.max_it = max_iterations
    
    return solver

def save_solution_for_visualization(
    w: Function,
    domain: Mesh,
    output_dir_path: str = "output_results",
    degree_vis_u: int = 1,
    degree_vis_p: int = 1,
    time_step: float = 0.0 # Optional: can be used for time-dependent problems
) -> Tuple[Function, Function]:
    """
    Interpolates the solution (velocity and pressure) into a lower-degree space
    and saves it to XDMF files for visualization in tools like Paraview.

    Args:
        w: The solution Function containing both velocity (P2) and pressure (P1).
        domain: The DolfinX Mesh object.
        output_dir_path: The directory path to save the output files.
        degree_vis_u: The polynomial degree for the velocity visualization space.
                      Defaults to 1 for compatibility with many viewers.
        degree_vis_p: The polynomial degree for the pressure visualization space.
                      Defaults to 1.
        time_step: The current time step value (for time-dependent problems).

    Returns:
        A tuple of the interpolated velocity and pressure Functions.
    """
    output_dir = Path(output_dir_path)
    output_dir.mkdir(exist_ok=True, parents=True)

    domain_cell = domain.basix_cell()
    V_u_vis = fem.functionspace(domain, element("Lagrange", domain_cell, degree_vis_u, shape=(domain.topology.dim,)))
    V_p_vis = fem.functionspace(domain, element("Lagrange", domain_cell, degree_vis_p))

    u_vis = Function(V_u_vis)
    p_vis = Function(V_p_vis)

    u_sol, p_sol = w.split()

    u_vis.interpolate(u_sol)
    p_vis.interpolate(p_sol)

    with XDMFFile(MPI.COMM_WORLD, output_dir / "velocity.xdmf", "w") as f_u:
        f_u.write_mesh(domain)
        u_vis.name = "velocity"
        f_u.write_function(u_vis, time_step)

    with XDMFFile(MPI.COMM_WORLD, output_dir / "pressure.xdmf", "w") as f_p:
        f_p.write_mesh(domain)
        p_vis.name = "pressure"
        f_p.write_function(p_vis, time_step)
        
    if MPI.COMM_WORLD.rank == 0:
        print(f"Interpolated solution saved to '{output_dir_path}' for visualization.")

    return u_vis, p_vis

def setup_pressure_nullspace(V_mixed: FunctionSpace, ksp_solver: PETSc.KSP, 
                             comm: MPI.Comm = MPI.COMM_WORLD,
                             pressure_index = 1):
    """
    Sets up a nullspace for the pressure component of a mixed function space,
    which is necessary when there is no Dirichlet boundary condition for pressure.
    This effectively sets the average pressure to zero.

    Args:
        V_mixed: The mixed function space (e.g., Taylor-Hood P2-P1).
        ksp_solver: The PETSc KSP object (the linear solver used by NewtonSolver).
        comm: The MPI communicator.
        pressure_index: The pressure index in the mixed function space.
    """
    V_p_collapsed, _ = V_mixed.sub(pressure_index).collapse()

    dofs_p = V_p_collapsed.dofmap.index_map.size_global * V_p_collapsed.dofmap.index_map_bs
    null_space_pressure = PETSc.Vec().create(comm)
    null_space_pressure.setSizes(dofs_p)
    null_space_pressure.setUp()
    
    # Fill the vector with 1.0 (representing a constant pressure field)
    local_size = null_space_pressure.getLocalSize()
    null_space_pressure.setArray(np.full(local_size, 1.0))    
    null_space_pressure.assemble()
    nsp = PETSc.NullSpace() 
    nsp.create(constant=True, comm=comm, vectors=[null_space_pressure])
    
    A_mat, P_mat = ksp_solver.getOperators() 
    A_mat.setNullSpace(nsp) 
    P_mat.setNullSpace(nsp) 

    if comm.rank == 0:
        print("Pressure nullspace successfully set for the KSP solver's matrices (A and P).")

def solve_adaptive_continuation(problem: NonlinearProblem, 
                                solver: NewtonSolver, 
                                w: Function,
                                nu: fem.Constant,
                                nu_target: float, 
                                nu_start: float = 1.0,
                                init_step: float = 0.1,
                                min_step: float = 0.01) -> bool:
    """
    Solve a nonlinear problem using adaptive continuation in the viscosity parameter `nu`.

    This function gradually reduces `nu` from `nu_start` towards `nu_target`, solving the nonlinear
    problem at each step using a Newton solver. If a step fails to converge, the continuation step 
    size is halved. If a step succeeds, the step size is doubled to accelerate the continuation.
    The function aborts if the step size falls below `min_step`.

    Args:
        problem: The nonlinear variational problem to be solved.
        solver: The Newton solver used to solve the problem.
        w: The solution function to be updated.
        nu: A DOLFINx constant representing the viscosity parameter in the problem.
        nu_target: The final target value of the viscosity.
        nu_start: The starting value of the viscosity (default is 1.0).
        init_step: The initial relative step size used in continuation (default is 0.1).
        min_step: The minimum allowed relative step size (default is 0.01).

    Returns:
        A boolean that states True if the method converged and False if not.
    Raises
    ------
    RuntimeError
        If the Newton solver fails to converge and the step size becomes too small.
    """
    cm = problem.L.mesh.comm
    nu.value = nu_start
    current_step = init_step
    old_nu = nu.value

    while nu.value > nu_target:
        trial_nu = old_nu * (1 - current_step)
        nu.value = max(trial_nu, nu_target)

        # Attempt to solve with current nu
        
        if cm.rank == 0:
            print(f"Trying nu = {nu.value:.2e}, step = {current_step:.2e} ...")
            sys.stdout.flush()
        try:
            its, converged = solver.solve(w)
        except:
            converged = False
            its = solver.max_it

        if cm.rank == 0:
            print(f"Converged: {converged}, Newton iterations: {its}")
            print("----------")
            sys.stdout.flush()

        if converged:
            # Success: increase step size, update reference value
            current_step *= 2.0
            old_nu = trial_nu
        else:
            # Failure: reduce step size
            current_step *= 0.5
            if current_step < min_step:
                if cm.rank == 0:
                    print("Adaptive continuation method failed due to too small step size.")
                    sys.stdout.flush()

                return False
                
    return True

def solve_with_continuation(problem: NonlinearProblem, 
                            solver: NewtonSolver, 
                            w: Function,
                            nu_fem_constant: fem.Constant, 
                            nu_target: float, 
                            num_steps: int = 1,
                            nu_start: float = None):
    """
    Iteratively solves a nonlinear problem using a continuation method with respect to viscosity. 
    from a start value to a target value over a specified number of steps.
    This approach helps in achieving convergence for challenging problems at low
    viscosities (high Reynolds numbers) by starting from an easier, higher viscosity state.

    Args:
        problem: The dolfinx.fem.petsc.NonlinearProblem object representing the system to solve.
        solver: The dolfinx.fem.petsc.NewtonSolver instance configured to solve the problem.
        w: The dolfinx.fem.Function object where the solution will be stored.
        nu_fem_constant: A dolfinx.fem.Constant object representing the viscosity.
        nu_target: The target viscosity value to be reached at the end.
        num_steps: The number of steps for the continuation.
                                If 1, a direct solution attempt at nu_target is made.
        nu_start: The starting viscosity for the continuation. If None and
                  num_steps > 1, the current value of nu_fem_constant
                  will be used as the starting point. If num_steps == 1,
                  this parameter is ignored.

    Returns:
        tuple: A tuple containing:
               - bool: True if all continuation steps converged successfully, False otherwise.
               - int: The total number of Newton iterations performed across all steps.
    """
    cm = problem.L.mesh.comm

    if num_steps < 1:
        raise ValueError("num_steps must be at least 1.")

    if nu_start is None:
        if num_steps > 1:
            current_nu_value = nu_fem_constant.value
        else: # one step
            current_nu_value = nu_target
    else:
        current_nu_value = nu_start

    if num_steps > 1:
        nu_factor = (nu_target / current_nu_value)**(1.0 / (num_steps - 1))
    else:
        nu_factor = 1.0

    all_converged = True
    total_iterations = 0

    if cm.rank == 0:
        print(f"Starting solver. Target nu: {nu_target:.2e}, Steps: {num_steps}")
        sys.stdout.flush()
        if num_steps > 1:
            print(f"  Continuation from nu_start: {current_nu_value:.2e}")
            sys.stdout.flush()

    for i in range(num_steps):
        if i == num_steps - 1:
            nu_fem_constant.value = nu_target
        else:
            nu_fem_constant.value = current_nu_value
        
        if cm.rank == 0:
            print(f"\n--- Solving Step {i+1}/{num_steps}: nu = {nu_fem_constant.value:.2e} ---")
            sys.stdout.flush()

        (num_its, converged) = solver.solve(w)

        total_iterations += num_its

        if cm.rank == 0:
            print(f"  Iterations: {num_its}, Converged: {converged}")
            sys.stdout.flush()

        if not converged:
            if cm.rank == 0:
                print(f"  Solver step {i+1} did not converge. Aborting.")
                sys.stdout.flush()
            all_converged = False
            break 
        
        if i < num_steps - 1:
            current_nu_value *= nu_factor
            
    if cm.rank == 0:
        if all_converged:
            print(f"\nSolver finished successfully. Final nu = {nu_fem_constant.value:.2e}")
            sys.stdout.flush()
        else:
            print(f"\nSolver procedure aborted due to non-convergence at an intermediate step.")
            sys.stdout.flush()

    return all_converged, total_iterations


if __name__ == "__main__":
    x_lim = 20.0
    y_lim = 10.0
    x_res = 32
    y_res = 16
    
    domain = create_2d_mesh_rect(x_lim, y_lim, x_res, y_res, mesh.CellType.quadrilateral)
    V = create_2d_th_functionspace(domain)
    
    w = fem.Function(V)
    w.x.array[:] = 0.0
    # rho = fem.Constant(domain, ScalarType(1.2041))  # kg/m³
    nu = fem.Constant(domain, ScalarType(14.9e-6))  # m²/s

    F, J = define_ns_ufl_forms(w, V, nu)
    
    def no_slip_bdry(x: np.ndarray) -> np.ndarray:
        return np.logical_or(np.isclose(x[1], 0), np.isclose(x[1], y_lim))
    
    def inflow(x: np.ndarray) -> np.ndarray:
        return np.isclose(x[0], 0)

    def outflow(x: np.ndarray) -> np.ndarray:
        return np.isclose(x[0], x_lim)

    def inflow_velocity_profile(x: np.ndarray) -> np.ndarray:
        ux_val = 4.0 * (y_lim - x[1]) * x[1] / y_lim**2
        return np.stack((ux_val, np.zeros_like(x[1])))
    
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
    solver = setup_newton_solver(problem, max_iterations=1000, ksp_type=KSP.Type.GMRES, 
                                 pc_type=PETSc.PC.Type.LU)
    (num_its, converged) = solver.solve(w)

    u_vis, ü_vis = save_solution_for_visualization(w, domain, output_dir_path="./output_results")
