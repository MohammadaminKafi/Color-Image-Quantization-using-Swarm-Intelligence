# Quantize an image based on the given centroids

# import libraries
import numpy as np
import cv2

# global parameters
population_size = 10
num_centroids = 16
kmeans_iters = 1
max_cycles = 10

# best solution filenames
ABC_best_filename = f"ABC_best_P{population_size}_K{num_centroids}_C{max_cycles}.txt"
SMO_best_filename = f"SMO_best_P{population_size}_K{num_centroids}_C{max_cycles}.txt"

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

# read the image
image_path = "./lena.png"
image = cv2.imread(image_path)
print("Image loaded successfully!")

# quantize the image
ABC_quantized_image = np.zeros(image.shape)
SMO_quantized_image = np.zeros(image.shape)
for i in range(image.shape[0]):
    for j in range(image.shape[1]):
        ABC_quantized_image[i, j] = ABC_best_solution[np.argmin(np.linalg.norm(image[i, j] - ABC_best_solution, axis=1))]
        SMO_quantized_image[i, j] = SMO_best_solution[np.argmin(np.linalg.norm(image[i, j] - SMO_best_solution, axis=1))]
ABC_quantized_image = ABC_quantized_image.astype(np.uint8)
SMO_quantized_image = SMO_quantized_image.astype(np.uint8)

# save the quantized images
ABC_quantized_image_path = f"./quantized_images/ABC_quantized_P{population_size}_K{num_centroids}_C{max_cycles}_{image_path[2:]}"
SMO_quantized_image_path = f"./quantized_images/SMO_quantized_P{population_size}_K{num_centroids}_C{max_cycles}_{image_path[2:]}"
cv2.imwrite(ABC_quantized_image_path, ABC_quantized_image)
cv2.imwrite(SMO_quantized_image_path, SMO_quantized_image)
print("Quantized images saved successfully!")
