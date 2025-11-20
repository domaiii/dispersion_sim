import gmsh
import pygmsh

from pathlib import Path

L = 20.0
H = 10.0

resolution = 0.04 * H

c1 = [0.2 * L, 0.7 * H, 0]
r1 = 0.15 * H

c2 = [0.75 * L, 0.5 * H, 0]
r2 = 0.15 * H

c3 = [0.3 * L, 0.4 * H, 0]
r3 = 0.1 * H

c4 = [0.4 * L, 0.6 * H, 0]
r4 = 0.07 * H

c5 = [0.9 * L, 0.2 * H, 0]
r5 = 0.1 * H

meshpath = Path(__file__).parent
msh_file = meshpath / "mesh.msh"

geometry = pygmsh.geo.Geometry()
model = geometry.__enter__()
circle1 = model.add_circle(c1, r1, mesh_size=resolution)
circle2 = model.add_circle(c2, r2, mesh_size=resolution)
circle3 = model.add_circle(c3, r3, mesh_size=resolution)
circle4 = model.add_circle(c4, r4, mesh_size=resolution)
circle5 = model.add_circle(c5, r5, mesh_size=resolution)

points = [
    model.add_point((0, 0, 0), mesh_size=resolution),
    model.add_point((L, 0, 0), mesh_size=resolution),
    model.add_point((L, H/2, 0), mesh_size=resolution),
    model.add_point((L, H, 0), mesh_size=resolution),
    model.add_point((0, H, 0), mesh_size=resolution),
    model.add_point((0, H/2, 0), mesh_size=resolution),
]

boundary_lines = [
    model.add_line(points[i], points[i + 1]) for i in range(-1, len(points) - 1)
]

loop = model.add_curve_loop(boundary_lines)
plane_surface = model.add_plane_surface(loop, holes=[circle1.curve_loop, circle2.curve_loop,
                                                     circle3.curve_loop, circle4.curve_loop,
                                                     circle5.curve_loop])

model.synchronize()

volume_marker = 6
model.add_physical([plane_surface], "Volume")
model.add_physical([boundary_lines[5], boundary_lines[0]], "Inflow")
model.add_physical([boundary_lines[2], boundary_lines[3]], "Outflow")
model.add_physical([boundary_lines[1], boundary_lines[4]], "Walls")
model.add_physical(circle1.curve_loop.curves +
                   circle2.curve_loop.curves +
                   circle3.curve_loop.curves +
                   circle4.curve_loop.curves +
                   circle5.curve_loop.curves, "Obstacles")

geometry.generate_mesh(dim=2)
gmsh.write(str(msh_file))
gmsh.clear()
geometry.__exit__()