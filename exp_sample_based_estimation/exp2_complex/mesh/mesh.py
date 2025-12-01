import gmsh
import pygmsh

from pathlib import Path

L = 20.0
H = 10.0

resolution = 0.04 * H

r1_xmin = 0.1 * L
r1_xmax = 0.15 * L
r1_ymin = 0.5 * H
r1_ymax = 0.8 * H

r2_xmin = 0.75 * L
r2_xmax = 0.85 * L
r2_ymin = 0.55 * H
r2_ymax = 0.7 * H

r3_xmin = 0.45 * L
r3_xmax = 0.55 * L
r3_ymin = 0.3 * H
r3_ymax = 0.45 * H

c1 = [0.3 * L, 0.25 * H, 0]
r1 = 0.15 * H

c2 = [0.4 * L, 0.8 * H, 0]
r2 = 0.07 * H

c3 = [0.9 * L, 0.2 * H, 0]
r3 = 0.1 * H

meshpath = Path(__file__).parent
msh_file = meshpath / "mesh.msh"

geometry = pygmsh.geo.Geometry()
model = geometry.__enter__()
rect1 = model.add_rectangle(r1_xmin, r1_xmax, r1_ymin, r1_ymax, 0.0, mesh_size=resolution)
rect2 = model.add_rectangle(r2_xmin, r2_xmax, r2_ymin, r2_ymax, 0.0, mesh_size=resolution)
rect3 = model.add_rectangle(r3_xmin, r3_xmax, r3_ymin, r3_ymax, 0.0, mesh_size=resolution)

circle1 = model.add_circle(c1, r1, mesh_size=resolution)
circle2 = model.add_circle(c2, r2, mesh_size=resolution)
circle3 = model.add_circle(c3, r3, mesh_size=resolution)

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
                                                     circle3.curve_loop, rect1.curve_loop,
                                                     rect2.curve_loop, rect3.curve_loop])

model.synchronize()

volume_marker = 6
model.add_physical([plane_surface], "Volume")
model.add_physical([boundary_lines[5], boundary_lines[0]], "Inflow")
model.add_physical([boundary_lines[2], boundary_lines[3]], "Outflow")
model.add_physical([boundary_lines[1], boundary_lines[4]], "Walls")
model.add_physical(circle1.curve_loop.curves + 
                   circle2.curve_loop.curves + 
                   circle3.curve_loop.curves + 
                   rect1.curve_loop.curves + 
                   rect2.curve_loop.curves +
                   rect3.curve_loop.curves, "Obstacles")

geometry.generate_mesh(dim=2)
gmsh.write(str(msh_file))
gmsh.clear()
geometry.__exit__()