import meshio
import gmsh
import pygmsh

resolution = 0.02

L = 2.0
H = 1.0

c1 = [0.15, 0.75, 0]
r1 = 0.1

c2 = [1.7, 0.5, 0]
r2 = 0.2

geometry = pygmsh.geo.Geometry()
model = geometry.__enter__()
circle1 = model.add_circle(c1, r1, mesh_size=resolution)
circle2 = model.add_circle(c2, r2, mesh_size=resolution)

points = [
    model.add_point((0, 0, 0), mesh_size=3*resolution),
    model.add_point((L, 0, 0), mesh_size=resolution),
    model.add_point((L, H/2, 0), mesh_size=resolution),
    model.add_point((L, H, 0), mesh_size=3*resolution),
    model.add_point((0, H, 0), mesh_size=resolution),
    model.add_point((0, H/2, 0), mesh_size=resolution),
]

boundary_lines = [
    model.add_line(points[i], points[i + 1]) for i in range(-1, len(points) - 1)
]

loop = model.add_curve_loop(boundary_lines)
plane_surface = model.add_plane_surface(loop, holes=[circle1.curve_loop, circle2.curve_loop])

model.synchronize()

volume_marker = 6
model.add_physical([plane_surface], "Volume")
model.add_physical([boundary_lines[5]], "Inflow")
model.add_physical([boundary_lines[2]], "Outflow")
model.add_physical([boundary_lines[0], boundary_lines[1], 
                    boundary_lines[3], boundary_lines[4]], "Walls")
model.add_physical(circle1.curve_loop.curves, "Obstacle1")
model.add_physical(circle2.curve_loop.curves, "Obstacle2")

geometry.generate_mesh(dim=2)
gmsh.write("2D_circular_obstacles.msh")
gmsh.clear()
geometry.__exit__()
