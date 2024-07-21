import numpy as np
from sklearn.cluster import KMeans
import time

def initialize_population(image, P, K):
    """
    Initialize the population with random centroids. As mentioned in the paper, the initial centroids are randomly selected from the image.

    Parameters:
    image (ndarray): The flattened image array.
    P (int): The population size.
    K (int): The number of clusters (centroids).

    Returns:
    list: A list of initial centroids for each solution in the population.
    """
    population = []
    for _ in range(P):
        # Randomly select K pixels from the image as initial centroids
        centroids = image[np.random.choice(image.shape[0], K, replace=False)] 
        population.append(centroids)
    return population

def apply_kmeans(centroids, image, K, n):
    """
    Refine the centroids using K-means clustering.

    Parameters:
    centroids (ndarray): Initial centroids.
    image (ndarray): The flattened image array.
    K (int): The number of clusters.
    n (int): The number of iterations.

    Returns:
    ndarray: The refined centroids.
    """
    kmeans = KMeans(n_clusters=K, init=centroids, n_init=1, max_iter=n)
    kmeans.fit(image)
    # round the centroids to the nearest integer
    centroids = np.round(kmeans.cluster_centers_).clip(0, 255)
    return centroids

def evaluate_solution(centroids, image):
    """
    Evaluate the solution by computing the Mean Squared Error (MSE).

    Parameters:
    centroids (ndarray): The centroids of the clusters.
    image (ndarray): The flattened image array.

    Returns:
    float: The MSE of the solution.
    """
    # Assign each pixel to the nearest centroid
    labels = np.argmin(np.linalg.norm(image[:, None] - centroids[None, :], axis=2), axis=1)
    # Calculate the MSE
    mse = np.mean([np.linalg.norm(image[i] - centroids[labels[i]])**2 for i in range(image.shape[0])])
    return mse

def combine_solution(centroids, neighbor_centroids):
    """
    Combine the centriods of the first solution by a random coefficient with the centroids of the second solution.

    Parameters:
    centroids (ndarray): The centroids of the first solution.
    neighbor_centroids (ndarray): The centroids of the second solution.

    Returns:
    ndarray: The combined solution.
    """
    # Generate a random coefficient for each dimension
    alpha = np.random.rand(centroids.shape[1])
    # Combine the centroids
    new_centroids = alpha * centroids + (1 - alpha) * neighbor_centroids
    return new_centroids
    

def generate_new_solution(image, K):
    """
    Generate a new solution from the given image.

    Parameters:
    image (ndarray): The flattened image array.
    K (int): The number of clusters.

    Returns:
    ndarray: The new solution (modified centroids).
    """
    # Randomly select K pixels from the image as new centroids
    new_centroids = image[np.random.choice(image.shape[0], K, replace=False)] 
    return new_centroids

def ABC_algorithm(image, P=10, K=16, unseccusful_trial_limit=5, kmeans_iters=3, max_cycles=100, verbose=False):
    # initialize log
    log = []

    # unravel the image
    image = image.reshape(-1, 3)

    # initialize population
    population = initialize_population(image, P, K)
    unseccusful_trials = np.zeros(P)

    # evaluate solutions
    mse_values = [evaluate_solution(solution, image) for solution in population]

    # iterate for a maximum number of cycles
    for _ in range(max_cycles):
        start_time = time.time()

        if verbose:
            print("Cycle:", _)
            print("Best MSE:", np.min(mse_values))

        # employed bees phase
        if verbose:
            print("Employed bees phase")
        for i in range(P):
            # apply K-Means
            population[i] = apply_kmeans(population[i], image, K, kmeans_iters)
            mse_values[i] = evaluate_solution(population[i], image)
            # combine the solution with a neighbor
            neighbor_index = np.random.choice([j for j in range(P) if j != i])
            neighbor_centroids = population[neighbor_index]
            new_centroids = combine_solution(population[i], neighbor_centroids)
            # evaluate the new solution
            new_mse = evaluate_solution(new_centroids, image)
            # update the solution if the new solution is better
            if new_mse < mse_values[i]:
                population[i] = new_centroids
                mse_values[i] = new_mse

        # onlooker bees phase
        if verbose:
            print("Onlooker bees phase")
        for i in range(P):
            # select a solution based on the probability
            sum_mse = np.sum(mse_values)
            probabilities = [mse / sum_mse for mse in mse_values]
            selected_index = np.random.choice(range(P), p=probabilities)
            # apply K-Means
            population[selected_index] = apply_kmeans(population[selected_index], image, K, kmeans_iters)
            mse_values[selected_index] = evaluate_solution(population[selected_index], image)
            # combine the solution with a neighbor
            neighbor_index = np.random.choice([j for j in range(P) if j != selected_index])
            neighbor_centroids = population[neighbor_index]
            new_centroids = combine_solution(population[selected_index], neighbor_centroids)
            # evaluate the new solution
            new_mse = evaluate_solution(new_centroids, image)
            # update the solution if the new solution is better
            if new_mse < mse_values[selected_index]:
                population[selected_index] = new_centroids
                mse_values[selected_index] = new_mse
            else:
                unseccusful_trials[selected_index] += 1

        # scout bees phase
        if verbose:
            print("Scout bees phase")
        for i in range(P):
            if unseccusful_trials[i] >= unseccusful_trial_limit:
                # generate a new solution
                population[i] = generate_new_solution(image, K)
                mse_values[i] = evaluate_solution(population[i], image)
                unseccusful_trials[i] = 0

        end_time = time.time()
        duration = end_time - start_time
        if verbose:
            print("Duration:", duration)

        # log the best solution
        best_index = np.argmin(mse_values)
        log.append((population[best_index], mse_values[best_index], duration))

    # return the best solution
    best_index = np.argmin(mse_values)
    return population[best_index], mse_values[best_index], log