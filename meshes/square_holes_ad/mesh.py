import gmsh
import pygmsh
from pathlib import Path

resolution = 0.01

L = 1.0
H = 1.0

c1 = [0.15, 0.6, 0]
r1 = 0.07

c2 = [0.75, 0.25, 0] 
r2 = 0.1

meshpath = Path(__file__).parent
msh_file = meshpath / "mesh.msh"

geometry = pygmsh.geo.Geometry()
model = geometry.__enter__()

# Add circles as holes first
circle1 = model.add_circle(c1, r1, mesh_size=resolution)
circle2 = model.add_circle(c2, r2, mesh_size=resolution)

# Define the outer rectangle
rectangle = model.add_rectangle(0, L, 0, H, 0, mesh_size=resolution)

# Get the corner points of the rectangle from the rectangle object
p0 = rectangle.points[0]
p1 = rectangle.points[1]
p2 = rectangle.points[2] 
p3 = rectangle.points[3] 

inlet_y_start = H / 2 - 0.05 * H
inlet_y_end = H / 2 + 0.05 * H
p_inlet_start = model.add_point((0, inlet_y_start, 0), mesh_size=resolution)
p_inlet_end = model.add_point((0, inlet_y_end, 0), mesh_size=resolution)

# Create the specific lines for the outer boundary, splitting the left edge
line_bottom = model.add_line(p0, p1)
line_right = model.add_line(p1, p2)
line_top = model.add_line(p2, p3)

# The left edge is now split into three segments
line_left_top_segment = model.add_line(p3, p_inlet_end)
line_inlet_segment = model.add_line(p_inlet_end, p_inlet_start) # Inlet segment
line_left_bottom_segment = model.add_line(p_inlet_start, p0)

# Create the outer curve loop using these new lines in order
outer_loop = model.add_curve_loop([
    line_bottom,
    line_right,
    line_top,
    line_left_top_segment,
    line_inlet_segment,
    line_left_bottom_segment
])

plane_surface = model.add_plane_surface(outer_loop, holes=[circle1.curve_loop, circle2.curve_loop])

model.synchronize()

model.add_physical([plane_surface], "Volume")
model.add_physical([line_inlet_segment], "Inflow")
model.add_physical([line_bottom, line_right, line_top, line_left_top_segment, 
                    line_left_bottom_segment], "Walls")
model.add_physical(circle1.curve_loop.curves, "Obstacle1")
model.add_physical(circle2.curve_loop.curves, "Obstacle2")

geometry.generate_mesh(dim=2)
gmsh.write(str(msh_file))
gmsh.clear()
geometry.__exit__()