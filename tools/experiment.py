from dataclasses import dataclass
import numpy as np
from ufl import inner, dx
from dolfinx import fem, mesh
from tools.airflow_estimator import AirflowEstimator
from tools.gas_estimator import GasSourceEstimator


@dataclass
class SingleExperimentResult:
    """Result container for a single experiment."""

    # Source information
    true_location: np.ndarray
    est_location: np.ndarray

    # Fields
    f_true: fem.Function
    f_est: fem.Function
    u_true: fem.Function
    u_est: fem.Function
    c_true: fem.Function
    c_est: fem.Function

    # Measurement coords
    gas_sample_coords: np.ndarray
    wind_sample_coords: np.ndarray

    # ---------- Error metrics (methods) ----------
    def _L2(self, fun:fem.Function) -> float:
        return float(np.sqrt(fem.assemble_scalar(fem.form(inner(fun, fun) * dx))))
    
    def _rel_error_L2(self, ref: fem.Function, fun: fem.Function) -> float:
        abs_err = self._L2(ref - fun)
        true_norm = self._L2(ref)
        return np.abs(abs_err / true_norm) if true_norm > 1e-14 else 0.0
    
    def localization_error(self) -> float:
        """Euclidean distance between true and estimated source location."""
        return float(np.linalg.norm(self.true_location - self.est_location))

    def abs_plume_error(self) -> float:
        """L2 error of concentration plume."""
        return self._L2(self.c_est - self.c_true)

    def rel_plume_error(self) -> float:
        return self._rel_error_L2(self.c_true, self.c_est)
    
    def abs_wind_error(self) -> float:
        """L2 error of reconstructed wind field."""
        return self._L2(self.u_est - self.u_true)
    
    def rel_wind_error(self) -> float:
        return self._rel_error_L2(self.u_true, self.u_est)
    
    def abs_source_error(self) -> float:
        """L2 error of source field."""
        return self._L2(self.f_est - self.f_true)




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
        
        self.use_true_wind = False

    @classmethod
    def with_true_wind(cls,
                    air_est,
                    gas_est,
                    source_location,
                    sigma_factor,
                    p_gas,
                    gas_seed,
                    gamma_reg=1e-2,
                    amplitude=10.0):
        """
        Build a SingleExperiment instance that skips wind reconstruction
        and instead injects the true wind.
        """
        obj = cls(
            air_est=air_est,
            gas_est=gas_est,
            source_location=source_location,
            sigma_factor=sigma_factor,
            p_wind=0,          # unused
            p_gas=p_gas,
            wind_seed=0,       # unused
            gas_seed=gas_seed,
            gamma_reg=gamma_reg,
            amplitude=amplitude,
        )
        obj.use_true_wind = True
        return obj

    def run_L1(self, verbose=True):
        return self._run(reg="L1", verbose=verbose)

    def run_L2(self, verbose=True):
        return self._run(reg="L2", verbose=verbose)
    
    # def _run(self, reg: str, verbose=True):
    #     air = self.air_est
    #     gas = self.gas_est

    #     # ---------------- Wind reconstruction ----------------
    #     if not self.use_true_wind:
    #         air.reset_random_measurements(self.p_wind, self.wind_seed)
    #         u_est = air.solve(maxit=5).sub(0).collapse()
    #     else:
    #         u_est = air.ground_truth.sub(0).collapse()
        
    #     u_true = air.ground_truth.sub(0).collapse()

    #     # ---------------- Gas reconstruction ----------------
    #     gas.reset_wind(u_est)

    #     sigma = self.sigma_factor * gas.Lx
    #     gas.set_true_gaussian_source(self.x0, self.y0, sigma, amplitude=self.amplitude)
    #     f_true = gas.f_true

    #     gas.reset_random_measurements(self.p_gas, seed=self.gas_seed)

    #     if reg == "L1":
    #         f_est = gas.solve_L1(gamma_reg=self.gamma_reg, verbose=verbose)
    #     elif reg == "L2":
    #         f_est = gas.solve_L2(gamma_reg=self.gamma_reg, verbose=verbose)
    #     else:
    #         raise ValueError(f"Unknown regularization type: {reg}")

    #     c_est = gas.c_est

    #     # ---------------- Save measurement coords ----------------
    #     gas_coords = gas.get_measurement_coordinates()
    #     if self.use_true_wind:
    #         wind_coords = None
    #     else:
    #         wind_coords = air.get_measurement_coordinates()

    #     # ---------------- Compute true gas dispersion ----------------
    #     gas.reset_wind(u_true)
    #     c_true = gas.dispersion_for_true_source()        

    #     # ---------------- Error metrics ----------------
    #     coords = gas.scalar_space.tabulate_dof_coordinates()
    #     true_loc = coords[np.argmax(f_true.x.array)]
    #     est_loc = coords[np.argmax(f_est.x.array)]

    #     # ---------------- Build result object ----------------
    #     return SingleExperimentResult(
    #         true_loc, est_loc,
    #         f_true.copy(), f_est.copy(),
    #         u_true.copy(), u_est.copy(),
    #         c_true.copy(), c_est.copy(),
    #         gas_coords, wind_coords
    #     )

    def _run(self, reg: str, verbose=True):
        air = self.air_est
        gas = self.gas_est

        # ----------------------------------------------------
        # 1) TRUE WIND + TRUE SOURCE → TRUE PLUME (Ground Truth)
        # ----------------------------------------------------
        u_true = air.ground_truth.sub(0).collapse()

        gas.reset_wind(u_true)

        sigma = self.sigma_factor * gas.Lx
        gas.set_true_gaussian_source(self.x0, self.y0, sigma, amplitude=self.amplitude)

        # Ground truth fields (copy AFTER setting true source)
        f_true = gas.f_true.copy()
        c_true = gas.dispersion_for_true_source().copy()

        # ----------------------------------------------------
        # 2) Wind Reconstruction (if not oracle case)
        # ----------------------------------------------------
        if not self.use_true_wind:
            air.reset_random_measurements(self.p_wind, self.wind_seed)
            u_est = air.solve(maxit=5).sub(0).collapse()
        else:
            u_est = u_true

        # ----------------------------------------------------
        # 3) Gas Reconstruction USING estimated wind
        # ----------------------------------------------------
        gas.reset_wind(u_est)
        gas.reset_random_measurements(self.p_gas, seed=self.gas_seed)

        if reg == "L1":
            f_est = gas.solve_L1(gamma_reg=self.gamma_reg, verbose=verbose)
        elif reg == "L2":
            f_est = gas.solve_L2(gamma_reg=self.gamma_reg, verbose=verbose)
        else:
            raise ValueError(f"Unknown regularization type: {reg}")

        c_est = gas.c_est.copy()

        # ----------------------------------------------------
        # 4) Measurement coordinates
        # ----------------------------------------------------
        gas_coords = gas.get_measurement_coordinates()
        wind_coords = None if self.use_true_wind else air.get_measurement_coordinates()

        # ----------------------------------------------------
        # 5) Source locations from DOF arrays
        # ----------------------------------------------------
        coords = gas.scalar_space.tabulate_dof_coordinates()
        true_loc = coords[np.argmax(f_true.x.array)]
        est_loc  = coords[np.argmax(f_est.x.array)]

        # ----------------------------------------------------
        # 6) Build result object
        # ----------------------------------------------------
        return SingleExperimentResult(
            true_location=true_loc,
            est_location=est_loc,
            f_true=f_true.copy(),
            f_est=f_est.copy(),
            u_true=u_true.copy(),
            u_est=u_est.copy(),
            c_true=c_true.copy(),
            c_est=c_est.copy(),
            gas_sample_coords=gas_coords,
            wind_sample_coords=wind_coords
        )