from pathlib import Path

import dolfinx.io as dio
from basix.ufl import element
from dolfinx import fem
from mpi4py import MPI

from tools.csv_utilities import load_from_3Dcsv
from tools.visualizer import MatplotlibVisualizer2D

SLICE_HEIGHT = 0.025
Z_TOL = 1e-1

CASE_STYLES = {
    '10x6 Appartment': {
        51: {'label': 'Walls', 'color': 'black', 'linewidth': 1.3},
        52: {'label': 'Inflow', 'color': 'tab:cyan', 'linewidth': 4.0},
        53: {'label': 'Inflow', 'color': 'tab:cyan', 'linewidth': 4.0},
        54: {'label': 'Outflow', 'color': 'tab:orange', 'linewidth': 4.0},
        55: {'label': 'Outflow', 'color': 'tab:orange', 'linewidth': 4.0},
    },
    '10x6 Labyrinth': {
        31: {'label': 'Walls', 'color': 'black', 'linewidth': 1.3},
        32: {'label': 'Inflow', 'color': 'tab:cyan', 'linewidth': 4.0},
        33: {'label': 'Inflow', 'color': 'tab:cyan', 'linewidth': 4.0},
        34: {'label': 'Outflow', 'color': 'tab:orange', 'linewidth': 4.0},
    },
    '10x6 Rectangular Obstacles': {
        184: {'label': 'Walls', 'color': 'black', 'linewidth': 1.3},
        185: {'label': 'Inflow', 'color': 'tab:cyan', 'linewidth': 4.0},
        186: {'label': 'Outflow', 'color': 'tab:orange', 'linewidth': 4.0},
    },
}


def plot_case(title: str, msh_path: Path, csv_path: Path, output_path: Path) -> None:
    domain, _, facet_tags = dio.gmshio.read_from_msh(str(msh_path), MPI.COMM_WORLD, gdim=2)
    elem_u = element('Lagrange', domain.basix_cell(), 1, shape=(domain.geometry.dim,))
    V = fem.functionspace(domain, elem_u)

    velocity = fem.Function(V)
    load_from_3Dcsv(csv_path, SLICE_HEIGHT, Z_TOL, velocity)

    vis = MatplotlibVisualizer2D(V)
    #vis.add_background_mesh(color='0.82', linewidth=0.22, alpha=0.7)
    vis.add_vector_field(title, velocity, stride=1, cmap='coolwarm')
    vis.add_boundary_facets(facet_tags, tag_styles=CASE_STYLES.get(title, {}))
    vis.show(title=title, filename=str(output_path), colorbar_label="Velocity magnitude (m/s)")


plot_case(
    title='10x6 Appartment',
    msh_path=Path('/app/csv_wind_data/10x6_appartment/appartment_2d.msh'),
    csv_path=Path('/app/csv_wind_data/10x6_appartment/cellcenters.csv'),
    output_path=Path('/app/wind_gt_10x6_appartment.png'),
)

plot_case(
    title='10x6 Labyrinth',
    msh_path=Path('/app/csv_wind_data/10x6_labyrinth/labyrinth_2d.msh'),
    csv_path=Path('/app/csv_wind_data/10x6_labyrinth/cellcenters.csv'),
    output_path=Path('/app/wind_gt_10x6_labyrinth.png'),
)

plot_case(
    title='10x6 Rectangular Obstacles',
    msh_path=Path('/app/csv_wind_data/10x6_multiple_obstacles/room_layer_2d.msh'),
    csv_path=Path('/app/csv_wind_data/10x6_multiple_obstacles/cellcenters.csv'),
    output_path=Path('/app/wind_gt_10x6_multiple_obstacles.png'),
)
