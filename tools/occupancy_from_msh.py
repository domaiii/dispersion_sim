from dataclasses import dataclass
import math
from pathlib import Path
import re
import argparse
import cv2
import meshio
import numpy as np

@dataclass(frozen=True)
class Config:
    msh_file: Path
    output_dir: Path
    resolution: float = 0.1
    outer_ring_cells: int = 1
    opening_tag_pattern: str = r"inlet|inflow|outlet|outflow"


@dataclass(frozen=True)
class RasterContext:
    points_xy: np.ndarray
    domain_xmin: float
    domain_xmax: float
    domain_ymin: float
    domain_ymax: float
    xmin: float
    xmax: float
    ymin: float
    ymax: float
    resolution: float
    width: int
    height: int

    @property
    def origin_y(self) -> float:
        return self.ymin

    def world_to_pixel(self, xy: np.ndarray) -> np.ndarray:
        px = np.floor((xy[:, 0] - self.xmin) / self.resolution).astype(np.int32)
        py = np.floor((self.ymax - xy[:, 1]) / self.resolution).astype(np.int32)
        pixels = np.column_stack([px, py])
        pixels[:, 0] = np.clip(pixels[:, 0], 0, self.width - 1)
        pixels[:, 1] = np.clip(pixels[:, 1], 0, self.height - 1)
        return pixels


def build_physical_tag_lookup(mesh: meshio.Mesh) -> dict[int, str]:
    return {int(data[0]): name for name, data in mesh.field_data.items()}


def get_cell_data_for_block(mesh: meshio.Mesh, key: str, block_index: int) -> np.ndarray | None:
    values = mesh.cell_data.get(key)
    if values is None or block_index >= len(values):
        return None
    return np.asarray(values[block_index])


def extract_triangles_and_tagged_lines(
    mesh: meshio.Mesh,
) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
    triangles: list[np.ndarray] = []
    tagged_line_blocks: list[tuple[np.ndarray, np.ndarray]] = []

    for block_index, cell_block in enumerate(mesh.cells):
        data = np.asarray(cell_block.data)

        if cell_block.type == "triangle":
            triangles.append(data)
            continue

        if cell_block.type != "line":
            continue

        physical_tags = get_cell_data_for_block(mesh, "gmsh:physical", block_index)
        if physical_tags is None:
            physical_tags = np.full(len(data), -1, dtype=int)
        tagged_line_blocks.append((data, physical_tags))

    if not triangles:
        raise ValueError("No triangle cells found in mesh.")

    return np.vstack(triangles), tagged_line_blocks


def compute_raster_context(points_xy: np.ndarray, resolution: float, outer_ring_cells: int) -> RasterContext:
    xmin = float(np.min(points_xy[:, 0]))
    xmax = float(np.max(points_xy[:, 0]))
    ymin = float(np.min(points_xy[:, 1]))
    ymax = float(np.max(points_xy[:, 1]))

    pad = outer_ring_cells * resolution
    xmin_map = xmin - pad
    xmax_map = xmax + pad
    ymin_map = ymin - pad
    ymax_map = ymax + pad

    width = int(math.ceil((xmax_map - xmin_map) / resolution))
    height = int(math.ceil((ymax_map - ymin_map) / resolution))

    return RasterContext(
        points_xy=points_xy,
        domain_xmin=xmin,
        domain_xmax=xmax,
        domain_ymin=ymin,
        domain_ymax=ymax,
        xmin=xmin_map,
        xmax=xmax_map,
        ymin=ymin_map,
        ymax=ymax_map,
        resolution=resolution,
        width=width,
        height=height,
    )


def points_in_triangle(px: np.ndarray, py: np.ndarray, tri_world: np.ndarray) -> np.ndarray:
    x1, y1 = tri_world[0]
    x2, y2 = tri_world[1]
    x3, y3 = tri_world[2]
    denom = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)
    if np.isclose(denom, 0.0):
        return np.zeros_like(px, dtype=bool)

    a = ((y2 - y3) * (px - x3) + (x3 - x2) * (py - y3)) / denom
    b = ((y3 - y1) * (px - x3) + (x1 - x3) * (py - y3)) / denom
    c = 1.0 - a - b
    eps = 1e-9
    return (a >= -eps) & (b >= -eps) & (c >= -eps)


def rasterize_free_space(image: np.ndarray, triangles: np.ndarray, ctx: RasterContext) -> None:
    for tri in triangles:
        tri_world = ctx.points_xy[tri]
        tri_pixels = ctx.world_to_pixel(tri_world)

        x0 = int(np.min(tri_pixels[:, 0]))
        x1 = int(np.max(tri_pixels[:, 0]))
        y0 = int(np.min(tri_pixels[:, 1]))
        y1 = int(np.max(tri_pixels[:, 1]))

        x_idx = np.arange(x0, x1 + 1)
        y_idx = np.arange(y0, y1 + 1)
        grid_x, grid_y = np.meshgrid(x_idx, y_idx)

        px = ctx.xmin + (grid_x + 0.5) * ctx.resolution
        py = ctx.ymax - (grid_y + 0.5) * ctx.resolution
        inside = points_in_triangle(px, py, tri_world)

        image[y0 : y1 + 1, x0 : x1 + 1][inside] = 255


def is_on_boundary(values: np.ndarray, target: float, resolution: float) -> bool:
    return bool(np.isclose(values, target, atol=resolution * 0.5).all())


def open_tagged_boundaries(
    image: np.ndarray,
    tagged_line_blocks: list[tuple[np.ndarray, np.ndarray]],
    tag_lookup: dict[int, str],
    opening_tag_pattern: re.Pattern[str],
    ctx: RasterContext,
) -> None:
    ring_cols_left = max(0, int(round((ctx.domain_xmin - ctx.xmin) / ctx.resolution)))
    ring_cols_right = max(0, int(round((ctx.xmax - ctx.domain_xmax) / ctx.resolution)))
    ring_rows_bottom = max(0, int(round((ctx.domain_ymin - ctx.ymin) / ctx.resolution)))
    ring_rows_top = max(0, int(round((ctx.ymax - ctx.domain_ymax) / ctx.resolution)))

    x_idx = np.arange(ctx.width)
    y_idx = np.arange(ctx.height)
    x_centers = ctx.xmin + (x_idx + 0.5) * ctx.resolution
    y_centers = ctx.ymax - (y_idx + 0.5) * ctx.resolution

    for lines, physical_tags in tagged_line_blocks:
        for segment, physical_id in zip(lines, physical_tags):
            tag_name = tag_lookup.get(int(physical_id), "")
            if not opening_tag_pattern.search(tag_name):
                continue

            segment_xy = ctx.points_xy[segment]
            x0, x1 = np.min(segment_xy[:, 0]), np.max(segment_xy[:, 0])
            y0, y1 = np.min(segment_xy[:, 1]), np.max(segment_xy[:, 1])

            if is_on_boundary(segment_xy[:, 0], ctx.domain_xmin, ctx.resolution):
                mask = (y_centers >= y0) & (y_centers < y1)
                if ring_cols_left > 0:
                    image[mask, :ring_cols_left] = 255
            elif is_on_boundary(segment_xy[:, 0], ctx.domain_xmax, ctx.resolution):
                mask = (y_centers >= y0) & (y_centers < y1)
                if ring_cols_right > 0:
                    image[mask, ctx.width - ring_cols_right :] = 255
            elif is_on_boundary(segment_xy[:, 1], ctx.domain_ymin, ctx.resolution):
                mask = (x_centers >= x0) & (x_centers < x1)
                if ring_rows_bottom > 0:
                    image[ctx.height - ring_rows_bottom :, mask] = 255
            elif is_on_boundary(segment_xy[:, 1], ctx.domain_ymax, ctx.resolution):
                mask = (x_centers >= x0) & (x_centers < x1)
                if ring_rows_top > 0:
                    image[:ring_rows_top, mask] = 255


def write_pgm(image: np.ndarray, path: Path) -> None:
    image_p2 = (image > 0).astype(np.uint8)
    height, width = image_p2.shape

    with path.open("w", encoding="ascii") as f:
        f.write(f"P2\n{width} {height}\n1\n")
        for row in image_p2:
            f.write(" ".join(str(int(value)) for value in row))
            f.write("\n")


def write_yaml(path: Path, image_name: str, resolution: float, origin_x: float, origin_y: float) -> None:
    path.write_text(
        (
            f"image: {image_name}\n"
            f"resolution: {resolution}\n"
            f"origin: [{origin_x}, {origin_y}, 0.0]\n"
            f"occupied_thresh: 0.9\n"
            f"free_thresh: 0.1\n"
            f"negate: 0\n"
        ),
        encoding="utf-8",
    )


def generate_occupancy_map(config: Config) -> tuple[Path, Path, dict[int, str]]:
    config.output_dir.mkdir(parents=True, exist_ok=True)

    mesh = meshio.read(config.msh_file)
    tag_lookup = build_physical_tag_lookup(mesh)
    points_xy = np.asarray(mesh.points)[:, :2].astype(np.float64)
    triangles, tagged_line_blocks = extract_triangles_and_tagged_lines(mesh)
    ctx = compute_raster_context(points_xy, config.resolution, config.outer_ring_cells)

    image = np.zeros((ctx.height, ctx.width), dtype=np.uint8)
    rasterize_free_space(image, triangles, ctx)
    open_tagged_boundaries(
        image,
        tagged_line_blocks,
        tag_lookup,
        re.compile(config.opening_tag_pattern, re.IGNORECASE),
        ctx,
    )

    pgm_path = config.output_dir / "occupancy.pgm"
    yaml_path = config.output_dir / "occupancy.yaml"
    write_pgm(image, pgm_path)
    write_yaml(yaml_path, pgm_path.name, config.resolution, ctx.xmin, ctx.origin_y)

    return pgm_path, yaml_path, tag_lookup


def main() -> None:

    parser = argparse.ArgumentParser(
                        prog='OccupancyFromMesh',
                        description='Create an occupancy.pgm and respective occupancy.yaml from a 2D msh file.')
    parser.add_argument("meshfile", type=str, help="Path to the .msh file to be converted.")
    parser.add_argument("-r", "--resolution", type=float, default=0.1, help="Desired resolution.")
    parser.add_argument("-o", "--output_dir", type=str, default=None, help="Output directory. Default is the meshfile location.")
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()
    output_dir = Path(args.meshfile).parent if args.output_dir is None else Path(args.output_dir)

    pgm_path, yaml_path, tag_lookup = generate_occupancy_map(Config(msh_file=Path(args.meshfile),
                                                                    output_dir=output_dir,
                                                                    resolution=args.resolution))

    if args.verbose:
        print("Done.")
        print(f"PGM written to:  {pgm_path}")
        print(f"YAML written to: {yaml_path}")
        print("Physical tags found:")
        for tag_id, tag_name in sorted(tag_lookup.items()):
            print(f"  {tag_id}: {tag_name}")

if __name__ == "__main__":
    main()
