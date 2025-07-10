from dolfinx import fem
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.io import gmshio
from dolfinx.plot import vtk_mesh
from mpi4py import MPI
import numpy as np
import pyvista
import gmsh
from petsc4py.PETSc import ScalarType, KSP
from petsc4py import PETSc

from ns_helpers import *

x_lim = 500
y_lim = 250

refine_box_xmin = 0.0
refine_box_xmax = 50  # Refine up to
refine_box_ymin = 0.4 * y_lim  # Cover a bit more than the nozzle height
refine_box_ymax = 0.6 * y_lim
min_cell_size_in_box = 0.5 # cell size in the box 
max_cell_size_outside_box = 5 # coarse cell size outside the box

# Continuation method parameters
nu_target = 14.9e-3 # target viscosity
nu_start = 0.1
num_continuation_steps = 5

comm = MPI.COMM_WORLD
model_rank = 0
gdim = 2

if comm.rank == model_rank:
    gmsh.initialize()
    gmsh.model.occ.synchronize()

    p1 = gmsh.model.occ.addPoint(0, 0, 0, tag=1)
    p2 = gmsh.model.occ.addPoint(x_lim, 0, 0, tag=2)
    p3 = gmsh.model.occ.addPoint(x_lim, y_lim, 0, tag=3)
    p4 = gmsh.model.occ.addPoint(0, y_lim, 0, tag=4)

    l1 = gmsh.model.occ.addLine(p1, p2, tag=1) 
    l2 = gmsh.model.occ.addLine(p2, p3, tag=2) 
    l3 = gmsh.model.occ.addLine(p3, p4, tag=3) 
    l4 = gmsh.model.occ.addLine(p4, p1, tag=4) 

    cl1 = gmsh.model.occ.addCurveLoop([l1, l2, l3, l4], tag=1)
    fluid_surface = gmsh.model.occ.addPlaneSurface([cl1], tag=1)
    
    gmsh.model.occ.synchronize()

    fluid_domain_tag = 100 
    inflow_tag = 101       
    walls_tag = 102        
    outflow_tag = 103      

    gmsh.model.addPhysicalGroup(gdim, [fluid_surface], fluid_domain_tag, name="FluidDomain")
    gmsh.model.addPhysicalGroup(gdim - 1, [l4], inflow_tag, name="Inflow")
    gmsh.model.addPhysicalGroup(gdim - 1, [l1, l3], walls_tag, name="Walls")
    gmsh.model.addPhysicalGroup(gdim - 1, [l2], outflow_tag, name="Outflow")

    gmsh.model.mesh.field.add("Box", 1)
    gmsh.model.mesh.field.setNumber(1, "VIn", min_cell_size_in_box)
    gmsh.model.mesh.field.setNumber(1, "VOut", max_cell_size_outside_box)
    gmsh.model.mesh.field.setNumber(1, "XMin", refine_box_xmin)
    gmsh.model.mesh.field.setNumber(1, "XMax", refine_box_xmax)
    gmsh.model.mesh.field.setNumber(1, "YMin", refine_box_ymin)
    gmsh.model.mesh.field.setNumber(1, "YMax", refine_box_ymax)
    
    gmsh.model.mesh.field.setAsBackgroundMesh(1)

    gmsh.model.mesh.generate(gdim)

domain, cell_tags, facet_tags = gmshio.model_to_mesh(gmsh.model, comm, model_rank, gdim=gdim)

if comm.rank == model_rank:
    topology, cell_types, geometry = vtk_mesh(domain)
    pyvista_mesh = pyvista.UnstructuredGrid(topology, cell_types, geometry)

    plotter = pyvista.Plotter()
    plotter.add_mesh(pyvista_mesh, show_edges=True, color="lightblue")
    plotter.show()

if comm.rank == model_rank:
    gmsh.finalize()

V = create_2d_th_functionspace(domain)
w = fem.Function(V)
w.x.array[:] = 0.0

nu_fem_constant = fem.Constant(domain, ScalarType(nu_start)) 

F, J = define_ns_ufl_forms(w, V, nu_fem_constant)

# No-slip boundary condition for walls (u = (0, 0))
V0, _ = V.sub(0).collapse() 
walls_facets = facet_tags.indices[facet_tags.values == walls_tag]
dofs_walls_list = fem.locate_dofs_topological((V.sub(0), V0), domain.topology.dim - 1, walls_facets)
zero_velocity_func = fem.Function(V0)
zero_velocity_func.x.array[:] = 0.0
bc_no_slip = fem.dirichletbc(zero_velocity_func, dofs_walls_list, V.sub(0)) 

# Inflow boundary condition (parabolic profile for velocity)
def inflow_profile_parabolic_middle(x: np.ndarray) -> np.ndarray:
    y_start_nozzle = 0.45 * y_lim
    y_end_nozzle = 0.55 * y_lim
    nozzle_height = y_end_nozzle - y_start_nozzle 
    y_coords = x[1]
    
    ux_val = np.zeros_like(y_coords) 
    in_nozzle = np.logical_and(y_coords >= y_start_nozzle, y_coords <= y_end_nozzle)
    y_relative = y_coords[in_nozzle] - y_start_nozzle 
    U_max = 0.1
    ux_val[in_nozzle] = U_max * 4.0 * y_relative * (nozzle_height - y_relative) / nozzle_height**2
    
    return np.stack((ux_val, np.zeros_like(y_coords)))

inflow_facets = facet_tags.indices[facet_tags.values == inflow_tag]
dofs_v_inflow_list = fem.locate_dofs_topological((V.sub(0), V0), domain.topology.dim -1, inflow_facets)

inflow_profile_dolfinx_func = fem.Function(V0)
inflow_profile_dolfinx_func.interpolate(inflow_profile_parabolic_middle)
bc_in = fem.dirichletbc(inflow_profile_dolfinx_func, dofs_v_inflow_list, V.sub(0))

# Outflow boundary condition (zero pressure)
V1, _ = V.sub(1).collapse()
outflow_facets = facet_tags.indices[facet_tags.values == outflow_tag]
dofs_p_outflow_list = fem.locate_dofs_topological((V.sub(1), V1), domain.geometry.dim - 1, outflow_facets)
zero_pressure_func = fem.Function(V1) 
zero_pressure_func.x.array[:] = 0.0
bc_out = fem.dirichletbc(zero_pressure_func, dofs_p_outflow_list, V.sub(1))

bcs = [bc_in, bc_no_slip, bc_out]

problem = NonlinearProblem(F, w, bcs, J)
solver = setup_newton_solver(problem, max_iterations=20, 
                             ksp_type=KSP.Type.GMRES, 
                             pc_type=PETSc.PC.Type.HYPRE, 
                             tolerance=1e-7)
setup_pressure_nullspace(V, solver.krylov_solver)

all_converged, _ = solve_with_continuation(problem, solver, w, nu_fem_constant, 
                                           nu_target, num_continuation_steps, nu_start)

if all_converged:
    u_vis, p_vis = save_solution_for_visualization(w, domain, output_dir_path="./output_results")
else:
    if MPI.COMM_WORLD.rank == 0:
        print("Solution not saved as solver procedure did not complete successfully.")