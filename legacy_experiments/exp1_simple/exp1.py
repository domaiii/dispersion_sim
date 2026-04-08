from pathlib import Path
import numpy as np
import pandas as pd

from dolfinx import fem
from tools.airflow_estimator import AirflowEstimator
from tools.gas_estimator import GasSourceEstimator
from tools.visualizer import Visualizer2D
from tools.experiment import SingleExperiment, SingleExperimentResult, ErrorValue

# -------------------------------------------------------------
# 1. Load mesh + wind ground truth + gas estimator
# -------------------------------------------------------------

WDIR = Path(__file__).parent.resolve()

bpfile   = WDIR / "airflow.bp"
meshfile = WDIR / "mesh/mesh.msh"

air_new = AirflowEstimator.from_bp(bpfile, fun_name="velocity_H2", meshfile=meshfile)

air_new.set_weights(
    kin_v=1.5e-4,
    misfit=1.0,
    pde_err=1.0,
    reg=1e-3
)

air_new.set_zero_pressure_bc("Outflow")
air_new.set_no_slip_bc(["Walls", "Obstacles"])
gas_new = GasSourceEstimator(air_new.domain, D_phys=0.002)

# -------------------------------------------------------------
# 2. Averaging parameters
# -------------------------------------------------------------

wind_seeds = range(1,4)
gas_seeds  = range(1,4)

sigma = 0.01
amplitude = 10.0

source_locations = [(0.1 * gas_new.Lx, 0.3 * gas_new.Ly), 
                    #(0.1 * gas_new.Lx, 0.5 * gas_new.Ly), 
                    (0.15 * gas_new.Lx, 0.9 * gas_new.Ly), 
                    # (0.2 * gas_new.Lx, 0.43 * gas_new.Ly), 
                    (0.25 * gas_new.Lx, 0.3 * gas_new.Ly), 
                    (0.3 * gas_new.Lx, 0.2 * gas_new.Ly), 
                    # (0.3 * gas_new.Lx, 0.72 * gas_new.Ly), 
                    (0.4 * gas_new.Lx, 0.5 * gas_new.Ly), 
                    (0.15 * gas_new.Lx, 0.1 * gas_new.Ly), 
                    (0.45 * gas_new.Lx, 0.13 * gas_new.Ly), 
                    # (0.5 * gas_new.Lx, 0.4 * gas_new.Ly), 
                    (0.4 * gas_new.Lx, 0.85 * gas_new.Ly), 
                    (0.35 * gas_new.Lx, 0.8 * gas_new.Ly), 
                    (0.38 * gas_new.Lx, 0.3 * gas_new.Ly), 
                    (0.5 * gas_new.Lx, 0.75 * gas_new.Ly)]

# Visual sanity check
vis = Visualizer2D(gas_new.scalar_space)
vis.add_background_mesh()
vis.add_points(source_locations, color="red", size=16)
vis.show("Candidate Source Locations")

sample_sizes = [25, 50, 100, 200, 400, 800]

gas_gamma_reg = 1e-3

def run_experiment_grid(
        air_est, gas_est,
        source_locations,
        p_wind_list,
        p_gas_list,
        wind_seeds,
        gas_seeds,
        sigma_factor=0.01,
        gamma_reg=5e-2,
        amplitude=10.0,
        outfile="exp1.parquet"
    ):

    rows = []

    # Total = number of RECONSTRUCTED runs
    total = (
        len(p_wind_list)
        * len(p_gas_list)
        * len(source_locations)
        * len(gas_seeds)
        * (1 + len(wind_seeds))
    )

    counter = 0

    print(f"Starting experiment grid with {total} reconstructed-wind runs total.")
    print("Oracle-wind runs are performed once per (source, p_gas, gas_seed).\n")

    # --- MAIN GRID ---
    for p_wind in p_wind_list:
        print(f"\n=== Wind samples p_wind = {p_wind} ===")

        for p_gas in p_gas_list:
            print(f"  -> Gas samples p_gas = {p_gas}")

            for (x0, y0) in source_locations:
                print(f"     Source = ({x0:.2f}, {y0:.2f})")

                for g_seed in gas_seeds:

                    # --------------------------------------------------
                    # ORACLE WIND RUN (only once per g_seed)
                    # --------------------------------------------------
                    counter += 1
                    print(f"       [{counter:4d} / {total}] ORACLE run (gas_seed={g_seed})")

                    exp_oracle = SingleExperiment.with_true_wind(
                        air_est=air_est,
                        gas_est=gas_est,
                        source_location=(x0, y0),
                        sigma_factor=sigma_factor,
                        p_gas=p_gas,
                        gas_seed=g_seed,
                        gamma_reg=gamma_reg,
                        amplitude=amplitude,
                    )

                    result_oracle = exp_oracle.run_L1(verbose=False)

                    rows.append({
                        # Parameters
                        "p_wind": np.inf,
                        "p_gas": p_gas,
                        "source_x": x0,
                        "source_y": y0,
                        "wind_seed": -1,
                        "gas_seed": g_seed,

                        # Error metrics
                        "loc_error": result_oracle.localization_error(),
                        "plume_L2": result_oracle.plume_error().L2,
                        "plume_RMS": result_oracle.plume_error().RMS,
                        "normalized_plume_err_L2": result_oracle.plume_error_norm().L2,
                        "normalized_plume_err_RMS": result_oracle.plume_error_norm().RMS,

                        # no wind estimation -> no wind data

                        "source_L2" : result_oracle.source_error().L2,
                        "source_RMS" : result_oracle.source_error().RMS,
                        "wind_source": "true",
                    })

                    gas_est.reset()

                    # --------------------------------------------------
                    # RECONSTRUCTED WIND RUNS
                    # --------------------------------------------------
                    for w_seed in wind_seeds:

                        counter += 1
                        print(f"       [{counter:4d} / {total}] "
                              f"Reconstruct (w_seed={w_seed}, g_seed={g_seed})")

                        exp = SingleExperiment(
                            air_est=air_est,
                            gas_est=gas_est,
                            source_location=(x0, y0),
                            sigma_factor=sigma_factor,
                            p_wind=p_wind,
                            p_gas=p_gas,
                            wind_seed=w_seed,
                            gas_seed=g_seed,
                            gamma_reg=gamma_reg,
                            amplitude=amplitude
                        )

                        result = exp.run_L1(verbose=False)

                        rows.append({
                            # Parameters
                            "p_wind": p_wind,
                            "p_gas": p_gas,
                            "source_x": x0,
                            "source_y": y0,
                            "wind_seed": w_seed,
                            "gas_seed": g_seed,

                            # Plume error metrics
                            "loc_error": result.localization_error(),
                            "plume_L2": result.plume_error().L2,
                            "plume_RMS": result.plume_error().RMS,
                            "normalized_plume_err_L2": result.plume_error_norm().L2,
                            "normalized_plume_err_RMS": result.plume_error_norm().RMS,

                            # Wind error metrics
                            "wind_L2": result.wind_error().L2,
                            "wind_RMS": result.wind_error().RMS,
                            "rel_wind_L2": result.wind_error_rel(),
                            "angular_wind_err_L2": result.angular_wind_error().L2,
                            "angular_wind_err_RMS": result.angular_wind_error().RMS,
                            "magnitude_err_L2": result.magnitude_wind_error().L2,
                            "magnitude_err_RMS": result.magnitude_wind_error().RMS,

                            # Source error metrics
                            "source_L2" : result.source_error().L2,
                            "source_RMS" : result.source_error().RMS,

                            "wind_source": "reconstructed",
                        })

                        gas_est.reset()

    df = pd.DataFrame(rows)
    df.to_parquet(outfile, index=False)

    print(f"\nSaved results to {outfile}")
    print("Finished all experiments.\n")

    return df


run_experiment_grid(
    air_new, gas_new,
    source_locations,
    sample_sizes, sample_sizes,
    wind_seeds, gas_seeds,
    sigma, gas_gamma_reg
)
