import matplotlib.pyplot as plt
import numpy as np
import yaml
import pandas as pd
from dataclasses import dataclass
from pathlib import Path

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

@dataclass
class Measurement:
    x: float
    y: float
    u_x: float
    u_y: float


class Grid:
    def __init__(self, n_cols: int, n_rows: int, cell_size: float, origin=(0.0, 0.0)):
        self.n_cols = int(n_cols)
        self.n_rows = int(n_rows)
        self.cell_size = float(cell_size)

        self.origin = origin
        x = origin[0] + (np.arange(self.n_cols) + 0.5) * self.cell_size
        y = origin[1] + (np.arange(self.n_rows) + 0.5) * self.cell_size
        self.xx, self.yy = np.meshgrid(x, y)
        self.points = np.column_stack([self.xx.ravel(), self.yy.ravel()])

        self.free_mask = np.ones_like(self.xx, dtype=bool)

        self.measurements: list[Measurement] = []

    @classmethod
    def from_occupancy_yaml(cls, yaml_path: str | Path):
        yaml_path = Path(yaml_path).resolve()
        with yaml_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        image_path = yaml_path.parent / data["image"]
        image = cls._read_pgm(image_path)
        image = np.flipud(image)

        resolution = float(data["resolution"])
        origin = data.get("origin", [0.0, 0.0, 0.0])
        origin_xy = (float(origin[0]), float(origin[1]))
        free_thresh = float(data.get("free_thresh"))
        occupied_thresh = float(data.get("occupied_thresh"))
        negate = int(data.get("negate"))

        occupancy = image.astype(float) / 255.0 if negate else (255.0 - image.astype(float)) / 255.0

        free_mask = occupancy < free_thresh
        unknown_mask = (occupancy >= free_thresh) & (occupancy <= occupied_thresh)
        free_mask[unknown_mask] = False

        n_rows, n_cols = image.shape
        grid = cls(n_cols=n_cols, n_rows=n_rows, cell_size=resolution, origin=origin_xy)
        grid.free_mask = free_mask
        return grid

    def gaussian_kernel(self, x: float, y: float, x0: float, y0: float, sigma: float) -> np.ndarray:
        dist_sq = (x - x0) ** 2 + (y - y0) ** 2
        return np.exp(-dist_sq / (2.0 * sigma**2))

    def add_csv_measurements(self, path: str | Path, count: int | None = None):
        samples_csv = Path(path).resolve(strict=True)
        df = pd.read_csv(samples_csv)
        if count is not None:
            count = int(count)
            if count < 1:
                raise ValueError(f"count must be at least 1, got {count}.")
            if count > len(df):
                raise ValueError(
                    f"Requested {count} measurements from {samples_csv.name}, but file only contains {len(df)} rows."
                )
            df = df.iloc[:count].copy()

        required = ["x", "y", "wind_x", "wind_y"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(
                f"Missing required columns in {samples_csv.name}: {missing}. "
                f"Expected at least {required}."
            )
        if len(df) == 0:
            raise ValueError(f"No sample rows found in {samples_csv}")

        for row in df.itertuples():
            self.add_measurement(row.x, row.y, row.wind_x, row.wind_y)

    def add_measurement(self, x, y, u_x, u_y):

        col_idx = int(np.floor((x - self.origin[0]) / self.cell_size))
        row_idx = int(np.floor((y - self.origin[1]) / self.cell_size))

        if col_idx < 0 or row_idx < 0 or col_idx >= self.n_cols or row_idx >= self.n_rows:
            raise ValueError(f"Measurement at ({x}, {y}) is out of bounds of the rectangular domain.")
        
        elif not self.free_mask[row_idx, col_idx] :
            print(f"Ignoring measurement at ({x}, {y}). Not within free domain area.")

        else:
            self.measurements.append(Measurement(x, y, u_x, u_y))

    def show_grid(self, output_path: str = "grid.png"):
        values = np.zeros_like(self.xx, dtype=float)
        values[~self.free_mask] = np.nan

        extent = (
            self.xx.min() - 0.5 * self.cell_size,
            self.xx.max() + 0.5 * self.cell_size,
            self.yy.min() - 0.5 * self.cell_size,
            self.yy.max() + 0.5 * self.cell_size,
        )

        fig, ax = plt.subplots(figsize=(8, 8), dpi=160)
        ax.imshow(values, origin="lower", extent=extent, cmap="Blues", vmin=0.0, vmax=1.0)

        occupied = np.ma.masked_where(self.free_mask, np.ones_like(values))
        ax.imshow(occupied, origin="lower", extent=extent, cmap="Greys", alpha=0.65, vmin=0.0, vmax=1.0)

        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("Grid occupancy")
        fig.tight_layout()
        fig.savefig(output_path)
        plt.close(fig)
        print(f"Saved {output_path}")

    def estimate_wind_field_gpr(self, length: float):
        x_m = [m.x for m in self.measurements]
        y_m = [m.y for m in self.measurements]
        ux_m = [m.u_x for m in self.measurements]
        uy_m = [m.u_y for m in self.measurements]

        kernel = 1 * RBF([length, length], length_scale_bounds=(1e-2, 1e2))
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=15)
        gp.fit(np.vstack([x_m, y_m]).T, np.vstack([ux_m, uy_m]).T)

        u_pred, u_std = gp.predict(self.points, return_std=True)
        
        u_field = u_pred[:, 0].reshape(self.xx.shape)
        v_field = u_pred[:, 1].reshape(self.yy.shape)

        u_field[~self.free_mask] = np.nan
        v_field[~self.free_mask] = np.nan

        return u_field, v_field

    def estimate_wind_field(self, sigma: float):
        sum_weights = np.zeros_like(self.xx)
        sum_u = np.zeros_like(self.xx)
        sum_v = np.zeros_like(self.xx)

        for m in self.measurements:
            w = self.gaussian_kernel(self.xx, self.yy, m.x, m.y, sigma)
            sum_weights += w
            sum_u += w * m.u_x
            sum_v += w * m.u_y

        u_field = np.divide(sum_u, sum_weights, out=np.zeros_like(sum_u), where=sum_weights > 1e-6)
        v_field = np.divide(sum_v, sum_weights, out=np.zeros_like(sum_v), where=sum_weights > 1e-6)
        u_field[~self.free_mask] = np.nan
        v_field[~self.free_mask] = np.nan
        
        return u_field, v_field
    
    def estimate_wind_field_gpr(self, length: float):
        x_m = np.array([m.x for m in self.measurements], dtype=float)
        y_m = np.array([m.y for m in self.measurements], dtype=float)
        ux_m = np.array([m.u_x for m in self.measurements], dtype=float)
        uy_m = np.array([m.u_y for m in self.measurements], dtype=float)

        kernel = 1.0 * RBF([length, length], length_scale_bounds=(1e-2, 1e2))
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
        gp.fit(np.vstack([x_m, y_m]).T, np.vstack([ux_m, uy_m]).T)

        u_pred, u_std = gp.predict(self.points, return_std=True)

        u_field = u_pred[:, 0].reshape(self.xx.shape)
        v_field = u_pred[:, 1].reshape(self.yy.shape)

        u_field[~self.free_mask] = np.nan
        v_field[~self.free_mask] = np.nan

        return u_field, v_field
    
    def plot_wind_field(self, u_field, v_field, title: str, output_path: str):
        fig, ax = plt.subplots(figsize=(8, 8))
        magnitude = np.sqrt(u_field**2 + v_field**2)

        extent = (
            self.xx.min() - 0.5 * self.cell_size,
            self.xx.max() + 0.5 * self.cell_size,
            self.yy.min() - 0.5 * self.cell_size,
            self.yy.max() + 0.5 * self.cell_size,
        )

        occupied = np.ma.masked_where(self.free_mask, np.ones_like(magnitude))
        ax.imshow(
            occupied,
            origin="lower",
            extent=extent,
            cmap="Greys",
            alpha=0.65,
            vmin=0.0,
            vmax=1.0,
        )

        quiv = ax.quiver(
            self.xx[self.free_mask],
            self.yy[self.free_mask],
            u_field[self.free_mask],
            v_field[self.free_mask],
            magnitude[self.free_mask],
            cmap="coolwarm",
            angles="xy",
            scale_units="xy",
            scale=None,
            pivot="tail",
            alpha=0.6,
        )
        fig.colorbar(quiv, ax=ax, pad=0.02, label="Wind speed")

        if self.measurements:
            m_x = [m.x for m in self.measurements]
            m_y = [m.y for m in self.measurements]
            ax.scatter(m_x, m_y, c="black", label="Measurements")

        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(title)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_path)
        plt.close(fig)
        
    @staticmethod
    def _read_pgm(path: str | Path) -> np.ndarray:
        path = Path(path)
        tokens = []
        with path.open("r", encoding="ascii") as f:
            for line in f:
                line = line.split("#", 1)[0].strip()
                if line:
                    tokens.extend(line.split())

        if not tokens or tokens[0] != "P2":
            raise ValueError(f"{path} is not an ASCII PGM (P2).")

        width = int(tokens[1])
        height = int(tokens[2])
        max_value = int(tokens[3])

        values = np.array(tokens[4:], dtype=float)
        expected = width * height
        if values.size != expected:
            raise ValueError(f"Expected {expected} pixels in {path}, found {values.size}.")

        values = np.rint(values * (255.0 / max_value)).astype(np.uint8)
        return values.reshape((height, width))

if __name__ == "__main__":
    grid = Grid.from_occupancy_yaml("/app/scenarios/10x6_labyrinth/geometry/occupancy.yaml")
    grid.add_csv_measurements("/app/scenarios/10x6_labyrinth/samples/sample_points_n400_seed8.csv", 50)
    
    u, v = grid.estimate_wind_field_gpr(10.0)

    grid.plot_wind_field(u, v, "Testing GPR", "gpr_test.png")

    
    



