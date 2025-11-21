from dataclasses import dataclass
import numpy as np
from dolfinx import fem, mesh
from tools.airflow_estimator import AirflowEstimator
from tools.gas_estimator import GasSourceEstimator


@dataclass
class SingleExperimentResult:
    """
    Result container for a single experiment.
    Includes source fields, wind fields, plume, and measurement coordinates.
    """

    # Source locations
    true_location: np.ndarray
    est_location: np.ndarray
    loc_error: float

    # Fields
    f_true: fem.Function
    f_est: fem.Function
    u_true: fem.Function
    u_est: fem.Function
    
    c_true: fem.Function
    c_est: fem.Function

    # Measurement coordinates
    gas_sample_coords: np.ndarray
    wind_sample_coords: np.ndarray


class SingleExperiment:
    """
    Minimal experiment wrapper for running a coupled wind-gas inverse problem.

    This class executes a *single* experiment consisting of:
        1. Wind field reconstruction from sparse velocity measurements.
        2. Gas source reconstruction from sparse concentration measurements.

    Parameters
    ----------
    air_est : AirflowEstimator
        A fully initialized airflow estimator. Boundary conditions
        must already be set before running the experiment.
    gas_est : GasSourceEstimator
        A gas source estimator instance. Its wind field will be replaced
        internally using the estimated wind from `air_est`.
    source_location : tuple[float, float]
        Coordinates (x0, y0) of the true Gaussian gas source.
    sigma_factor : float
        Gaussian width relative to domain size, i.e. `sigma = sigma_factor * Lx`.
    p_wind : int
        Number of wind measurement points to sample.
    p_gas : int
        Number of gas measurement points to sample.
    wind_seed : int
        Random seed for selecting wind measurement DOFs.
    gas_seed : int
        Random seed for selecting gas measurement DOFs.
    gamma_reg : float, optional
        L1 regularization weight for the gas source reconstruction (default: 1e-2).
    amplitude : float, optional
        Peak amplitude of the true Gaussian source (default: 10.0).

    Returns
    -------
    result: SingleExperimentResult
        Contains the data from the experiment.
    """

    def __init__(
        self,
        air_est: AirflowEstimator,
        gas_est: GasSourceEstimator,
        source_location: tuple[float, float],
        sigma_factor: float,
        p_wind: int,
        p_gas: int,
        wind_seed: int,
        gas_seed: int,
        gamma_reg: float = 1e-2,
        amplitude: float = 10.0,
    ):
        self.air_est = air_est
        self.gas_est = gas_est

        self.x0, self.y0 = source_location
        self.sigma_factor = sigma_factor
        self.amplitude = amplitude

        self.p_wind = p_wind
        self.p_gas = p_gas
        self.wind_seed = wind_seed
        self.gas_seed = gas_seed

        self.gamma_reg = gamma_reg


    def run(self):
        air = self.air_est
        gas = self.gas_est

        # ---------------- Wind reconstruction ----------------
        air.reset_random_measurements(self.p_wind, self.wind_seed)
        u_est = air.solve(maxit=5).sub(0).collapse()
        u_true = air.ground_truth.sub(0).collapse()

        # ---------------- Gas reconstruction ----------------
        gas.reset_wind(u_est)

        sigma = self.sigma_factor * gas.Lx
        gas.set_true_gaussian_source(self.x0, self.y0, sigma, amplitude=self.amplitude)
        f_true = gas.f_true

        c_true = gas.get_ground_truth_concentration()
        gas.reset_random_measurements(self.p_gas, seed=self.gas_seed)

        # source reconstruction
        f_est = gas.solve_L1(gamma_reg=self.gamma_reg, verbose=True)
        c_est = gas.c_est

        # ---------------- Error metrics ----------------
        coords = gas.scalar_space.tabulate_dof_coordinates()
        true_loc = coords[np.argmax(f_true.x.array)]
        est_loc = coords[np.argmax(f_est.x.array)]
        loc_error = float(np.linalg.norm(true_loc - est_loc))

        # ---------------- Save measurement coords ----------------
        gas_coords = gas.get_measurement_coordinates()
        wind_coords = air.get_measurement_coordinates()

        # ---------------- Build result object ----------------
        return SingleExperimentResult(
            true_loc, est_loc, loc_error,
            f_true, f_est,
            u_true, u_est,
            c_true, c_est,
            gas_coords, wind_coords
        )