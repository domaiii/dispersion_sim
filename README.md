# Sample-Based Wind and Gas Source Estimation with FEM

Inverse finite element workflows for airflow reconstruction, gas source estimation, and sample-based experiments (FEniCSx/DOLFINx).

## What this repo contains

- `tools/`: core building blocks (`AirflowEstimator`, `GasSourceEstimator`, visualization, experiment containers).
- `airflow_estimation/`: airflow reconstruction workflows and notebooks.
- `gas_estimation/`: gas inverse problem workflows.
- `exp_sample_based_estimation/`: experiment setups (currently only one) and evaluation scripts
- `meshes/`: reusable mesh assets.

## Quick start using Docker

1. Build and start:
   ```bash
   docker compose up --build
   ```
2. Run scripts inside the container shell in `/app`.


## Main example workflows

- Airflow estimation demo:
  ```bash
  /dolfinx-env/bin/python /app/airflow_estimation/air_estimation_example_workflow.py
  ```
- Sample-based experiment grid:
  ```bash
  /dolfinx-env/bin/python /app/exp_sample_based_estimation/exp1_simple/exp1.py
  ```
  Runtime can increase significantly with larger sample grids and multiple random seeds.

Both scripts write output images/data files to the current working directory (or experiment folder) since the docker environment is headless.

## Notes

- Most visualizations run in headless mode and save images instead of opening a window.
- FE Mesh `.msh` and simulation results `.bp` input paths in workflows are currently hard-coded to `/app/...` for container usage.
