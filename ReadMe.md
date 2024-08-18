# Color Image Quantization using Swarm Intelligence

This project implements color image quantization using swarm intelligence techniques. It includes the implementation of the Artificial Bee Colony (ABC) algorithm, based on the paper "Color Image Quantization: A Short Review and an Application with Artificial Bee Colony Algorithm," and a newly proposed and implemented method using Spider Monkey Optimization (SMO). Additionally, the project references a method based on Particle Swarm Optimization (PSO) from the paper "A Color Image Quantization Algorithm Based on Particle Swarm."

## Project Structure

The project has the following directories:

1. `src`: Contains the source code.
2. `test_images`: Contains the test image `lena.png`.
3. `quantized_images`: Contains the quantized images generated by the algorithms.
4. `logs`: Contains log files of the best values found and the time taken by the algorithms.

### Directories and Files

#### 1. src

- `main.py`: Implements a comparison between the ABC and SMO methods.
- `ABC_utils.py`: Contains utility functions for the ABC algorithm.
- `SMO_utils.py`: Contains utility functions for the SMO algorithm.
- `analyzer.py`: Shows a comparison between implementations, loading its data from the `logs` directory.
- `quantizer.py`: Quantizes and saves the quantized images based on values in the `logs` directory.

#### 2. test_images

- `lena.png`: The test image used for quantization.

#### 3. quantized_images

Contains images quantized using different methods. The files are named as:
```
{algorithm_name}_quantized_P{algorithm_population_size}_K{number_of_centroids}_C{number_of_cycles}_{test_image_name}.png
```

#### 4. logs

Contains log files:
- Best values found by the algorithms, named as:
  ```
  {algorithm_name}_best_P{algorithm_population_size}_K{number_of_centroids}_C{number_of_cycles}.txt
  ```
- Log of the time taken by the algorithms to find the solution, named as:
  ```
  {algorithm_name}_log_P{algorithm_population_size}_K{number_of_centroids}_C{number_of_cycles}.csv
  ```

## Algorithm Implementation Details

### Artificial Bee Colony (ABC) Algorithm

The ABC algorithm mimics the foraging behavior of honey bees. The algorithm is divided into three phases: employed bee phase, onlooker bee phase, and scout bee phase. Each phase is responsible for exploring and exploiting the solution space to find optimal color centroids for quantizing the image.

#### Phases:
- **Employed Bee Phase:** Employed bees search for new food sources (solutions) around the current food source.
- **Onlooker Bee Phase:** Onlooker bees select food sources based on a probability related to the quality of the food source.
- **Scout Bee Phase:** Scouts search for new food sources randomly.

#### K-Means Integration:
Before each iteration, K-Means is applied to refine the centroids, ensuring that the local search is effective and that the centroids are better positioned.

### Spider Monkey Optimization (SMO)

The SMO algorithm is inspired by the foraging behavior of spider monkeys. It divides the population into groups and each group searches for food sources independently, allowing better exploration of the solution space.

#### Phases:
- **Updating Position Based on Local Leader:** Update each member's position using the local leader and a random neighbor. Accept the new position if it improves fitness.

- **Updating Position Based on Global Leader:** Update positions of selected members based on the global leader and a random neighbor. Accept improvements in fitness.

- **Learning Global Leader:** Identify the best solution as the global leader. Reset or increment the global leader counter (`GLC`) based on changes.

- **Learning Local Leader:** Identify the best solution in each group as the local leader. Reset or increment the local leader counter (`LLC`) based on changes.

- **Deciding Local Leader:** Reorganize the group if the local leader's counter exceeds a threshold. Update positions or generate new solutions, then re-evaluate the local leader.

- **Deciding Fission or Fusion:** Adjust the number of groups based on the global leader counter (`GLC`). Create new groups (fission) or merge all into one group (fusion), then reset `GLC`.


#### K-Means Integration:
Similar to ABC, K-Means is applied before each iteration in SMO to refine the centroids, ensuring that the solutions converge more effectively.

## Global Variables and Their Functionality

### Global Variables in `main.py`

- `verbose`: If set to `True`, the program will print detailed information during execution.
- `image_path`: The path to the input image.
- `population_size`: The number of solutions (food sources or monkeys) in the population.
- `num_centroids`: The number of color centroids for quantization.
- `kmeans_iters`: The number of iterations for the K-means algorithm used within the swarm algorithms.
- `max_cycles`: The maximum number of cycles for the swarm algorithms.

### ABC Algorithm Parameters

- `unseccusful_trial_limit`: The limit of unsuccessful trials before an employed bee becomes a scout.

### SMO Algorithm Parameters

- `GLL`: The number of iterations for the Global Leader Learning phase.
- `LLL`: The number of iterations for the Local Leader Learning phase.
- `MG`: The number of monkey groups.
- `PR`: The perturbation rate controlling how solutions are perturbed.

## Examples

Below is an example of the original image and the quantized image generated using the Spider Monkey Optimization (SMO) algorithm with specific parameters.

### Original Image
![Original Image](test_images/lena.png)

### Quantized Image using ABC
![SMO Quantized Image](quantized_images/ABC_quantized_P10_K16_C10_lena.png)

### Quantized Image using SMO
![SMO Quantized Image](quantized_images/SMO_quantized_P10_K16_C10_lena.png)

In this example, the following parameters were used for the SMO algorithm:
- **Population Size:** 10
- **Number of Centroids (K):** 16
- **Number of Cycles (C):** 10

As shown, the quantized image reduces the number of colors while maintaining the overall appearance of the original image, demonstrating the effectiveness of the SMO algorithm for color image quantization.

## References

- [Color Image Quantization: A Short Review and an Application with Artificial Bee Colony Algorithm](https://avesis.erciyes.edu.tr/yayin/c249fedb-9cb5-45a4-89f5-2d559ddbd69f/color-image-quantization-a-short-review-and-an-application-with-artificial-bee-colony-algorithm/document.pdf)
- [A Color Image Quantization Algorithm Based on Particle Swarm](https://informatica.si/index.php/informatica/article/download/40/34)
