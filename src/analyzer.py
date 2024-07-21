# Analize and plot the data related to each ABC and SMO log

# import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# global parameters
population_size = 10
num_centroids = 16
kmeans_iters = 1
max_cycles = 10

# log filenames
ABC_log_filename = f"./logs/ABC_log_P{population_size}_K{num_centroids}_C{max_cycles}.csv"
SMO_log_filename = f"./logs/SMO_log_P{population_size}_K{num_centroids}_C{max_cycles}.csv"

# read the log files
ABC_log = pd.read_csv(ABC_log_filename)
SMO_log = pd.read_csv(SMO_log_filename)

# plot the MSE vs Cycle
plt.figure()
plt.plot(ABC_log["Cycle"], ABC_log["MSE"], label="ABC")
plt.plot(SMO_log["Cycle"], SMO_log["MSE"], label="SMO")
plt.xlabel("Cycle")
plt.xticks(np.arange(0, max_cycles+1, 1))
plt.ylabel("MSE")
plt.title("MSE vs Cycle")
plt.legend()
plt.ylim(200, 250)  # Focus on MSE values from 250 to 300

# plot the Duration vs Cycle
plt.figure()
plt.plot(ABC_log["Cycle"], ABC_log["Duration"], label="ABC")
plt.plot(SMO_log["Cycle"], SMO_log["Duration"], label="SMO")
plt.xlabel("Cycle")
plt.xticks(np.arange(0, max_cycles+1, 1))
plt.ylabel("Duration")
plt.title("Duration vs Cycle")
plt.legend()

# show the plots
plt.show()

# best solution filenames
ABC_best_filename = f"./logs/ABC_best_P{population_size}_K{num_centroids}_C{max_cycles}.txt"
SMO_best_filename = f"./logs/SMO_best_P{population_size}_K{num_centroids}_C{max_cycles}.txt"

# read the best solutions
ABC_best_solution = np.zeros((num_centroids, 3))
with open(ABC_best_filename, "r") as f:
    i = 0
    for line in f:
        line = line[1:-3].replace(" ", "").split(".")
        line = [float(value) for value in line]
        ABC_best_solution[i] = line
        i += 1
SMO_best_solution = np.zeros((num_centroids, 3))
with open(SMO_best_filename, "r") as f:
    i = 0
    for line in f:
        line = line[1:-3].replace(" ", "").split(".")
        line = [float(value) for value in line]
        SMO_best_solution[i] = line
        i += 1
        

# plot the best centroids
plt.figure()
plt.scatter(ABC_best_solution[:, 0], ABC_best_solution[:, 1], label="ABC")
plt.scatter(SMO_best_solution[:, 0], SMO_best_solution[:, 1], label="SMO")
plt.xlabel("R")
plt.ylabel("G")
plt.title("Best Centroids")
plt.legend()

# show the plot
plt.show()
