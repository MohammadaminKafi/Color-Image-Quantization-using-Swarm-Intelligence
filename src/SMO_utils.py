import numpy as np
from sklearn.cluster import KMeans
import time

def initialize_population(image, P, K):
    """
    Initialize the population with random centroids. The initial centroids are randomly selected from the image.

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

def update_position_based_on_leader(centroids, leader, neighbor_centroids, updating_dimensions):
    """
    Update the position of the solution based on the local leader.

    Parameters:
    centroids (ndarray): The centroids of the clusters.
    leader (ndarray): The local or global leader centroids.
    neighbor_centroids (ndarray): The centroids of the neighboring solutions.
    updating_dimensions: A binary array determining which dimensions to update

    Returns:
    ndarray: The updated centroids.
    """
    # Calculate the new position based on the local leader
    new_centroids = centroids.copy()
    for i in range(3):
        if updating_dimensions[i]:
            new_centroids[:, i] = (centroids[:, i] + np.random.uniform(0, 1) * (leader[:, i] - centroids[:, i]) + np.random.uniform(-1, 1) * (neighbor_centroids[:, i] - centroids[:, i]))
    # round and clip the centroids
    new_centroids = np.round(new_centroids).clip(0, 255)
    return new_centroids  

def decide_position_based_on_leader(centroids, local_leader, global_leader, updating_dimensions):
    """
    Decide the position of the solution based on the local and global leaders.

    Parameters:
    centroids (ndarray): The centroids of the clusters.
    local_leader (ndarray): The local leader centroids.
    global_leader (ndarray): The global leader centroids.
    updating_dimensions: A binary array determining which dimensions to update

    Returns:
    ndarray: The updated centroids.
    """
    # Calculate the new position based on the local and global leaders
    new_centroids = centroids.copy()
    for i in range(3):
        if updating_dimensions[i]:
            new_centroids[:, i] = (centroids[:, i] + np.random.uniform(0, 1) * (global_leader[:, i] - centroids[:, i]) + np.random.uniform(0, 1) * (centroids[:, i] - local_leader[:, i]))
    # round and clip the centroids
    new_centroids = np.round(new_centroids).clip(0, 255)
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

def SMO_algorithm(image, P, K, max_cycles, GLL, LLL, MG, PR, kmeans_iters=3, verbose=False):
    # initialize log
    log = []

    # unravel the image
    image = image.reshape(-1, 3)

    # initialize the population
    population = initialize_population(image, P, K)

    # evaluate the initial population
    fitness = [evaluate_solution(centroids, image) for centroids in population]

    # identify global and local leaders
    number_of_groups = 1
    group_size = P // number_of_groups
    global_leader_index = np.argmin(fitness)
    local_leader_index = [global_leader_index]

    GLC = 0
    LLC = [0]

    # iterate through the cycles
    for _ in range(max_cycles):
        start_time = time.time()

        if verbose:
            print("Cycle:", _)
            print("Best MSE:", np.min(fitness))

        # apply K-Means to each solution
        if verbose:
            print("0. Applying K-Means")
        for i in range(P):
            population[i] = apply_kmeans(population[i], image, K, 3)
            fitness[i] = evaluate_solution(population[i], image)

        # update position based on local leader
        if verbose:
            print("1. Updating position based on local leader")
        for i in range(P):
            member_group = i // group_size
            updating_dimensions = np.random.rand(3) < PR
            # choose a random neighbor from the group
            neighbor_index = np.random.choice([j for j in range(member_group * group_size, (member_group + 1) * group_size) if j != i])
            # update the position based on the local leader
            new_member = update_position_based_on_leader(population[i], population[local_leader_index[member_group]], population[neighbor_index], updating_dimensions)
            # evaluate the new member
            new_fitness = evaluate_solution(new_member, image)
            # update the member if the new position is better
            if new_fitness < fitness[i]:
                population[i] = new_member
                fitness[i] = new_fitness

        # update position based on global leader
        if verbose:
            print("2. Updating position based on global leader")
        total_fitness = sum(fitness)
        probabilities = [fitness[i] / total_fitness for i in range(P)]
        cnt = 0
        while cnt < group_size:
            for i in range(P):
                if np.random.rand() < probabilities[i]:
                    cnt += 1
                    updating_dimensions = np.random.rand(3) < PR
                    # choose a random neighbor from the group
                    neighbor_index = np.random.choice([j for j in range(P) if j != i])
                    # update the position based on the global leader
                    new_member = update_position_based_on_leader(population[i], population[global_leader_index], population[neighbor_index], updating_dimensions)
                    # evaluate the new member
                    new_fitness = evaluate_solution(new_member, image)
                    # update the member if the new position is better
                    if new_fitness < fitness[i]:
                        population[i] = new_member
                        fitness[i] = new_fitness

        # learn global leader
        if verbose:
            print("3. Learning global leader")
        new_global_leader_index = np.argmin(fitness)
        if new_global_leader_index != global_leader_index:
            global_leader_index = new_global_leader_index
            GLC = 0
        else:
            GLC += 1

        # learn local leader
        if verbose:
            print("4. Learning local leader")
        for i in range(number_of_groups):
            new_local_leader_index = np.argmin(fitness[i * group_size:(i + 1) * group_size])
            if new_local_leader_index != local_leader_index[i]:
                local_leader_index[i] = new_local_leader_index
                LLC[i] = 0
            else:
                LLC[i] += 1

        # decide local leader
        if verbose:
            print("5. Deciding local leader")
        for i in range(number_of_groups):
            if LLC[i] > LLL:
                LLC[i] = 0
                for j in range(i * group_size, (i + 1) * group_size):
                    if np.random.rand() > PR:
                        population[j] = generate_new_solution(image, K)
                        fitness[j] = evaluate_solution(population[j], image)
                    else:
                        updating_dimensions = np.random.rand(3) < PR
                        population[j] = decide_position_based_on_leader(population[j], population[local_leader_index[i]], population[global_leader_index], updating_dimensions)
                        fitness[j] = evaluate_solution(new_member, image)
                local_leader_index[i] = np.argmin(fitness[i * group_size:(i + 1) * group_size])

        # decide fission or fusion
        if verbose:
            print("6. Deciding fission or fusion")
        if GLC >= GLL:
            if number_of_groups < MG:
                number_of_groups += 1
                group_size = P // number_of_groups
                local_leader_index.append(global_leader_index)
                LLC = [0 for _ in range(number_of_groups)]
            else:
                number_of_groups = 1
                group_size = P // number_of_groups
                local_leader_index = [global_leader_index]
                LLC = [0]
            GLC = 0

        end_time = time.time()
        duration = end_time - start_time
        if verbose:
            print("Duration:", duration)

        # log the best solution
        best_index = np.argmin(fitness)
        log.append((population[best_index], fitness[best_index], duration))

    # return the best solution
    best_index = np.argmin(fitness)
    return population[best_index], fitness[best_index], log