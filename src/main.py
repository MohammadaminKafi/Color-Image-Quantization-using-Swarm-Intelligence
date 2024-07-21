# ABC and SMO implementation of CIQ problem 
# Based on the paper "Color Image Quantization: A Short Review and an Application with Artificial Bee Colony Algorithm" by C. Ozturk et al.
# The RGB version of the paper is implemented here

# import libraries
import numpy as np
import cv2
import matplotlib.pyplot as plt

# import functions
from ABC_utils import ABC_algorithm
from SMO_utils import SMO_algorithm

# global variables
verbose = True
image_path = "./lena.png"

# read the image
image = cv2.imread(image_path)
if verbose:
    print("Image loaded successfully!")
    print("Image shape:", image.shape)

# global parameters
population_size = 10
num_centroids = 16
kmeans_iters = 1
max_cycles = 10

# ABC algorithm's parameters
unseccusful_trial_limit = 5

# SMO algorithm's parameters
GLL = 4
LLL = 2
MG = 2
PR = 0.6

# run the ABC algorithm
ABC_best_solution, ABC_best_mse, ABC_log = ABC_algorithm(image, P=population_size, K=num_centroids, unseccusful_trial_limit=unseccusful_trial_limit, kmeans_iters=kmeans_iters, max_cycles=max_cycles, verbose=verbose)

# run the SMO algorithm
SMO_best_solution, SMO_best_mse, SMO_log = SMO_algorithm(image, P=population_size, K=num_centroids, max_cycles=max_cycles, GLL=GLL, LLL=LLL, MG=MG, PR=PR, kmeans_iters=kmeans_iters, verbose=verbose)

# results filenames
ABC_log_filename = f"./logs/ABC_log_P{population_size}_K{num_centroids}_C{max_cycles}.csv"
SMO_log_filename = f"./logs/SMO_log_P{population_size}_K{num_centroids}_C{max_cycles}.csv"
ABC_best_filename = f"./logs/ABC_best_P{population_size}_K{num_centroids}_C{max_cycles}.txt"
SMO_best_filename = f"./logs/SMO_best_P{population_size}_K{num_centroids}_C{max_cycles}.txt"

# save the log in a .csv file
with open(ABC_log_filename, "w") as f:
    f.write("Cycle,MSE,Duration\n")
    for i in range(len(ABC_log)):
        f.write(f"{i},{ABC_log[i][1]},{ABC_log[i][2]}\n")
with open(SMO_log_filename, "w") as f:
    f.write("Cycle,MSE,Duration\n")
    for i in range(len(SMO_log)):
        f.write(f"{i},{SMO_log[i][1]},{SMO_log[i][2]}\n")

# save best centroids in a .txt file
with open(ABC_best_filename, "w") as f:
    for centroid in ABC_best_solution:
        f.write(f"{centroid}\n")
with open(SMO_best_filename, "w") as f:
    for centroid in SMO_best_solution:
        f.write(f"{centroid}\n")