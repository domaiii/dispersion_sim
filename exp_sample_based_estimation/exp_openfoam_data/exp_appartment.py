
import glob, os
import gmsh
import numpy as np
import pandas as pd
import dolfinx.io as dio
import matplotlib.pyplot as plt
from basix.ufl import element
from dolfinx import fem
from mpi4py import MPI
from pathlib import Path
from tools import csv_utilities 
from tools.airflow_estimator import AirflowEstimator
from tools.gas_estimator import GasSourceEstimator
from tools.visualizer import MatplotlibVisualizer2D

meshfile = "/app/csv_wind_data/10x6_appartment/appartment_2d.msh"
csv_file = "/app/csv_wind_data/10x6_appartment/cellcenters.csv"

domain, _, facet_tags = dio.gmshio.read_from_msh(meshfile, MPI.COMM_WORLD, gdim=2)
elem_u = element("Lagrange", domain.basix_cell(), 2, shape=(domain.geometry.dim,))
V = fem.functionspace(domain, elem_u)

wind_gt = fem.Function(V)
csv_utilities.csv_to_function(csv_file, 0.025, 1e-2, wind_gt, max_xy_dist=0.2)
air_est = AirflowEstimator.from_domain(domain, facet_tags=facet_tags, meshfile=meshfile, ground_truth=wind_gt)
