from pathlib import Path
import numpy as np
import pandas as pd

from dolfinx import fem
from tools.airflow_estimator import AirflowEstimator
from tools.gas_estimator import GasSourceEstimator
from tools.visualizer import Visualizer2D
from tools.experiment import SingleExperiment, SingleExperimentResult

# -------------------------------------------------------------
# 1. Load mesh + wind ground truth + gas estimator
# -------------------------------------------------------------

WDIR = Path(__file__).parent.resolve()

bpfile   = WDIR / "airflow.bp"
meshfile = WDIR / "mesh/mesh.msh"

air_new = AirflowEstimator.from_file(bpfile, fun_name="velocity_H2", meshfile=meshfile)

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
                    # (0.38 * gas_new.Lx, 0.3 * gas_new.Ly), 
                    (0.5 * gas_new.Lx, 0.75 * gas_new.Ly)]

# Visual sanity check
vis = Visualizer2D(gas_new.scalar_space)
vis.add_background_mesh()
vis.add_points(source_locations, color="red", size=16)
vis.show("Candidate Source Locations")

sample_sizes = [25, 50, 100, 200, 400, 1000]

gas_gamma_reg = 1e-3

# to be deleted afterwards!
def debug_stopper(result, result_oracle, gas_est, x0, y0):
    """Debug only when localization error with reconstructed wind is better."""

    if False: # result.localization_error() >= result_oracle.localization_error():
        return  # nothing to debug

    # Extract points
    true_pt_rec   = result.true_location
    true_pt_oracle   = result.true_location
    reco_pt   = result.est_location
    oracle_pt = result_oracle.est_location  # estimated location with true wind

    print("\n\n===== DEBUG TRIGGERED (Localization Worse With Reconstructed Wind) =====")
    print(f"True location reco           : ({true_pt_rec[0]:.4f}, {true_pt_rec[1]:.4f})")
    print(f"True location oracle         : ({true_pt_oracle[0]:.4f}, {true_pt_oracle[1]:.4f})")
    print(f"Estimated (reconstructed)    : ({reco_pt[0]:.4f}, {reco_pt[1]:.4f})")
    print(f"Estimated (oracle true wind) : ({oracle_pt[0]:.4f}, {oracle_pt[1]:.4f})")
    print("")
    print(f"Oracle loc error      : {result_oracle.localization_error():.4f}")
    print(f"Reco-wind loc error   : {result.localization_error():.4f}")
    print("")
    print(f"Oracle plume error    : {result_oracle.abs_plume_error():.4f}")
    print(f"Reco plume error      : {result.abs_plume_error():.4f}")
    print("\nVisualization open — close windows and press ENTER to continue.\n")

    # === Plot reconstructed plume and the three locations ===
    vis = Visualizer2D(gas_est.scalar_space)
    vis.add_scalar_field("Estimated source term (reconstructed wind)", result.f_est, cmap="coolwarm")
    vis.add_points([reco_pt], label="reconstructed source max")
    vis.add_points([true_pt_rec], "green", label="true source max")

    vis.show("RECONSTRUCTED WIND")

    vis2 = Visualizer2D(gas_est.scalar_space)
    vis2.add_scalar_field("Estimated source term (with true wind)", result_oracle.f_est, cmap="coolwarm")
    vis2.add_points([oracle_pt],label="reconstructed source max")
    vis2.add_points([true_pt_rec], "green", label="true source max")
    vis2.show("TRUE WIND")

    # vis3 = Visualizer2D(gas_est.scalar_space)
    # vis3.add_scalar_field("Estimated plume (with true wind)", result_oracle.c_est, cmap="coolwarm")
    # vis3.add_points([oracle_pt])
    # vis3.show("DEBUG — Localization Failure")

    # vis4 = Visualizer2D(gas_est.scalar_space)
    # vis4.add_scalar_field("True plume (with true wind)", result_oracle.c_true, cmap="coolwarm")
    # vis4.add_points([oracle_pt])
    # vis4.show("DEBUG — Localization Failure")

    # vis4 = Visualizer2D(gas_est.scalar_space)
    # vis4.add_scalar_field("True plume (with recostructed wind)", result.c_true, cmap="coolwarm")
    # vis4.add_points([oracle_pt])
    # vis4.show("DEBUG — Localization Failure")


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
        outfile="test_bug_exp.parquet"
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
                        "p_wind": np.inf,
                        "p_gas": p_gas,
                        "source_x": x0,
                        "source_y": y0,
                        "wind_seed": -1,
                        "gas_seed": g_seed,
                        "loc_error": result_oracle.localization_error(),
                        "plume_L2": result_oracle.abs_plume_error(),
                        "rel_plume_L2": result_oracle.rel_plume_error(),
                        "wind_L2": 0.0,
                        "rel_wind_L2": 0.0,
                        "source_L2" : result_oracle.abs_source_error(),
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
                            "p_wind": p_wind,
                            "p_gas": p_gas,
                            "source_x": x0,
                            "source_y": y0,
                            "wind_seed": w_seed,
                            "gas_seed": g_seed,
                            "loc_error": result.localization_error(),
                            "plume_L2": result.abs_plume_error(),
                            "rel_plume_L2": result.rel_plume_error(),
                            "wind_L2": result.abs_wind_error(),
                            "rel_wind_L2": result.rel_wind_error(),
                            "source_L2" : result.abs_source_error(),
                            "wind_source": "reconstructed",
                        })

                        gas_est.reset()

                        #debug_stopper(result, result_oracle, gas_est, x0, y0)

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

