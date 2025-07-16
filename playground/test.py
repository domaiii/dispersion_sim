from mpi4py import MPI
import numpy as np

arr = np.random.rand(2,2)
print(f"I am {MPI.COMM_WORLD.rank}. My array is {arr}.")
