from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import gmsh
import numpy as np
import pyvista as pv

from mpi4py import MPI
from petsc4py import PETSc
from basix.ufl import element
from dolfinx import fem, io, plot
from dolfinx.fem import Constant, Function, form, assemble_scalar
from dolfinx.fem.petsc import (
    create_matrix,
    create_vector,
    assemble_matrix,
    assemble_vector,
    apply_lifting,
    set_bc,
)
from ufl import (
    CellDiameter,
    TestFunction,
    TrialFunction,
    div,
    dot,
    dx,
    grad,
    inner,
    lhs,
    nabla_grad,
    rhs,
    sqrt,
)


@dataclass
class IPCSConfig:
    meshfile: Path
    T: float = 3.0
    dt: float = 1.0 / 1000.0
    mu: float = 1e-5
    rho: float = 1.0
    inflow_name: str = "Inlet"
    outflow_name: str = "Outlet"
    wall_names: tuple[str, ...] = ("Walls",)
    t_ramp_up: float = 2.0
    v_max: float = 1.0
    tol_stationary: float = 1e-6
    enable_supg: bool = True
    supg_scale: float = 1.0


@dataclass
class IPCSResult:
    domain: object
    facet_tags: object
    velocity: Function
    pressure: Function
    l2_errors: list[float]
    converged_iteration: int | None
    name_to_id: dict[str, int]


def _physical_name_to_id(meshfile: Path) -> dict[str, int]:
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    try:
        gmsh.open(str(meshfile))
        groups = gmsh.model.getPhysicalGroups()
        return {gmsh.model.getPhysicalName(dim, tag): tag for (dim, tag) in groups}
    finally:
        gmsh.finalize()


class ParabolicInflow:
    def __init__(self, t: float, y_lim: float, t_ramp_up: float, v_max: float):
        self.t = t
        self.y_lim = y_lim
        self.t_ramp_up = t_ramp_up
        self.v_max = v_max

    def __call__(self, x):
        ramp = min(self.t / self.t_ramp_up, 1.0)
        scale = ramp * self.v_max
        v_x = scale * 4.0 * (x[1] / self.y_lim) * (1.0 - x[1] / self.y_lim)
        v_y = np.zeros_like(x[1])
        return np.stack((v_x, v_y))


def run_ipcs(cfg: IPCSConfig) -> IPCSResult:
    domain, _, facet_tags = io.gmshio.read_from_msh(cfg.meshfile, MPI.COMM_WORLD, gdim=2)
    name_to_id = _physical_name_to_id(cfg.meshfile)

    coords = domain.geometry.x
    y_lim = np.max(coords[:, 1])

    elem_u = element("Lagrange", domain.basix_cell(), 2, shape=(2,))
    elem_p = element("Lagrange", domain.basix_cell(), 1)
    V = fem.functionspace(domain, elem_u)
    Q = fem.functionspace(domain, elem_p)

    fdim = domain.topology.dim - 1
    t = 0.0
    num_steps = int(cfg.T / cfg.dt)

    k = Constant(domain, PETSc.ScalarType(cfg.dt))
    mu = Constant(domain, PETSc.ScalarType(cfg.mu))
    rho = Constant(domain, PETSc.ScalarType(cfg.rho))

    u_nonslip = np.array((0,) * domain.geometry.dim, dtype=PETSc.ScalarType)
    wall_facets = np.concatenate([facet_tags.find(name_to_id[n]) for n in cfg.wall_names])
    wall_dofs = fem.locate_dofs_topological(V, fdim, wall_facets)
    no_slip_bc = fem.dirichletbc(u_nonslip, wall_dofs, V)

    inflow_facets = facet_tags.find(name_to_id[cfg.inflow_name])
    inflow_dofs = fem.locate_dofs_topological(V, fdim, inflow_facets)
    inlet_velocity = ParabolicInflow(t=t, y_lim=y_lim, t_ramp_up=cfg.t_ramp_up, v_max=cfg.v_max)
    u_inlet = Function(V)
    u_inlet.interpolate(inlet_velocity)
    bc_in = fem.dirichletbc(u_inlet, inflow_dofs)

    outflow_facets = facet_tags.find(name_to_id[cfg.outflow_name])
    dofs_out = fem.locate_dofs_topological(Q, fdim, outflow_facets)
    bc_out = fem.dirichletbc(PETSc.ScalarType(0), dofs_out, Q)

    velocity_bcs = [bc_in, no_slip_bc]
    pressure_bcs = [bc_out]

    u = TrialFunction(V)
    v = TestFunction(V)
    p = TrialFunction(Q)
    q = TestFunction(Q)

    u_ = Function(V, name="u")
    u_s = Function(V, name="u_tentative")
    u_n = Function(V)
    u_n1 = Function(V)
    p_ = Function(Q, name="p")
    phi = Function(Q, name="phi")

    f = Constant(domain, PETSc.ScalarType((0, 0)))

    u_adv = 1.5 * u_n - 0.5 * u_n1
    u_mid = 0.5 * (u + u_n)

    F1 = rho / k * dot(u - u_n, v) * dx
    F1 += inner(dot(u_adv, nabla_grad(u_mid)), v) * dx
    F1 += mu * inner(grad(u_mid), grad(v)) * dx - dot(p_, div(v)) * dx
    F1 += dot(f, v) * dx

    if cfg.enable_supg:
        h = CellDiameter(domain)
        u_adv_norm = sqrt(dot(u_adv, u_adv) + PETSc.ScalarType(1e-12))
        tau = Constant(domain, PETSc.ScalarType(cfg.supg_scale)) / sqrt(
            (2.0 / k) ** 2 + (2.0 * u_adv_norm / h) ** 2 + (4.0 * mu / (rho * h * h)) ** 2
        )

        Rm = rho / k * (u - u_n)
        Rm += rho * dot(u_adv, nabla_grad(u_mid))
        Rm -= mu * div(grad(u_mid))
        Rm += grad(p_) - f

        F1 += tau * inner(Rm, dot(u_adv, nabla_grad(v))) * dx

    a1 = form(lhs(F1))
    L1 = form(rhs(F1))
    A1 = create_matrix(a1)
    b1 = create_vector(L1)

    a2 = form(dot(grad(p), grad(q)) * dx)
    L2 = form(-rho / k * dot(div(u_s), q) * dx)
    A2 = assemble_matrix(a2, bcs=pressure_bcs)
    A2.assemble()
    b2 = create_vector(L2)

    a3 = form(rho * dot(u, v) * dx)
    L3 = form(rho * dot(u_s, v) * dx - k * dot(nabla_grad(phi), v) * dx)
    A3 = assemble_matrix(a3)
    A3.assemble()
    b3 = create_vector(L3)

    solver1 = PETSc.KSP().create(domain.comm)
    solver1.setOperators(A1)
    solver1.setType(PETSc.KSP.Type.BCGS)
    solver1.getPC().setType(PETSc.PC.Type.JACOBI)

    solver2 = PETSc.KSP().create(domain.comm)
    solver2.setOperators(A2)
    solver2.setType(PETSc.KSP.Type.MINRES)
    pc2 = solver2.getPC()
    pc2.setType(PETSc.PC.Type.HYPRE)
    pc2.setHYPREType("boomeramg")

    solver3 = PETSc.KSP().create(domain.comm)
    solver3.setOperators(A3)
    solver3.setType(PETSc.KSP.Type.CG)
    solver3.getPC().setType(PETSc.PC.Type.SOR)

    num_dofs = V.dofmap.index_map.size_global * V.dofmap.index_map_bs
    u_diff = Function(V)
    l2_norm_squared = form(inner(u_diff, u_diff) * dx)

    l2_errors: list[float] = []
    it_conv: int | None = None

    for i in range(num_steps):
        t += cfg.dt
        inlet_velocity.t = t
        u_inlet.interpolate(inlet_velocity)

        A1.zeroEntries()
        assemble_matrix(A1, a1, bcs=velocity_bcs)
        A1.assemble()
        with b1.localForm() as loc:
            loc.set(0)
        assemble_vector(b1, L1)
        apply_lifting(b1, [a1], [velocity_bcs])
        b1.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        set_bc(b1, velocity_bcs)
        solver1.solve(b1, u_s.x.petsc_vec)
        u_s.x.scatter_forward()

        with b2.localForm() as loc:
            loc.set(0)
        assemble_vector(b2, L2)
        apply_lifting(b2, [a2], [pressure_bcs])
        b2.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        set_bc(b2, pressure_bcs)
        solver2.solve(b2, phi.x.petsc_vec)
        phi.x.scatter_forward()

        p_.x.petsc_vec.axpy(1, phi.x.petsc_vec)
        p_.x.scatter_forward()

        with b3.localForm() as loc:
            loc.set(0)
        assemble_vector(b3, L3)
        b3.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        solver3.solve(b3, u_.x.petsc_vec)
        u_.x.scatter_forward()

        u_diff.x.array[:] = u_.x.array - u_n.x.array
        l2_err = np.sqrt(assemble_scalar(l2_norm_squared)) / num_dofs
        l2_errors.append(l2_err)

        if t > cfg.t_ramp_up and l2_err < cfg.tol_stationary:
            it_conv = i
            break

        with (
            u_.x.petsc_vec.localForm() as loc_,
            u_n.x.petsc_vec.localForm() as loc_n,
            u_n1.x.petsc_vec.localForm() as loc_n1,
        ):
            loc_n.copy(loc_n1)
            loc_.copy(loc_n)

    return IPCSResult(
        domain=domain,
        facet_tags=facet_tags,
        velocity=u_n,
        pressure=p_,
        l2_errors=l2_errors,
        converged_iteration=it_conv,
        name_to_id=name_to_id,
    )


def plot_streamlines(result: IPCSResult, nx: int = 100, ny: int = 50, density: float = 1.6):
    from tools.visualizer import MatplotlibVisualizer2D

    vis = MatplotlibVisualizer2D(result.velocity.function_space)
    vis.add_background_mesh()
    vis.add_streamplot("Result IPCS", result.velocity, nx, ny, density)
    vis.show("IPCS Forward Simulation")


def plot_glyphs(result: IPCSResult):
    topology, cell_type, geom = plot.vtk_mesh(result.velocity.function_space)
    grid = pv.UnstructuredGrid(topology, cell_type, geom)
    wind2d = result.velocity.x.array.reshape(-1, 2)
    wind3d = np.hstack((wind2d, np.zeros((wind2d.shape[0], 1))))
    grid.point_data["wind_vectors"] = wind3d
    subset = grid.extract_points(np.arange(0, grid.n_points, 1), include_cells=False)
    glyphs = subset.glyph(orient="wind_vectors", scale="wind_vectors", factor=1.0)

    pl = pv.Plotter()
    pl.add_mesh(glyphs, scalar_bar_args={"title": "Velocity magnitude [m/s]"})
    pl.add_mesh(grid, color="k", opacity=0.3)
    pl.view_xy()
    pl.zoom_camera(1.3)
    pl.show_axes()
    pl.show()


def write_adios_checkpoint(path: Path, result: IPCSResult, function_name: str = "velocity_H2"):
    import adios4dolfinx

    adios4dolfinx.write_mesh(path, result.domain)
    adios4dolfinx.write_meshtags(path, result.domain, result.facet_tags, meshtag_name="facet_tags")
    adios4dolfinx.write_function(path, result.velocity, name=function_name)


def write_vtx(path: Path, result: IPCSResult, t: float = 0.0):
    from dolfinx.io import VTXWriter

    with VTXWriter(result.domain.comm, str(path), [result.velocity], engine="BP4") as vtx:
        vtx.write(t)
