from mpi4py import MPI
import random
import cv2
import math
import time
import numpy as np

# Set random seeds for reproducibility
random.seed(7)

# Function to get initial centroids for k-means
def get_initial_centroids(X, k):
    sample_points_ids = random.sample(range(0, len(X)), k)
    centroids = [tuple(X[id]) for id in sample_points_ids]
    unique_centroids = list(set(centroids))

    while len(unique_centroids) < k:
        new_sample_points_ids = random.sample(range(0, len(X)), k - len(unique_centroids))
        new_centroids = [tuple(X[id]) for id in new_sample_points_ids]
        unique_centroids = list(set(unique_centroids + new_centroids))

    return unique_centroids

# Function to compute Euclidean distance between two matrices
def get_euclidean_distance(A_matrix, B_matrix):
    distances = [[math.sqrt(sum((float(a_i) - float(b_i)) ** 2 for a_i, b_i in zip(a, b))) for b in B_matrix] for a in A_matrix]
    return distances

# Function to assign points to clusters based on centroids
def get_clusters(X, centroids, distance_mesuring_method):
    k = len(centroids)
    clusters = {i: [] for i in range(k)}

    distance_matrix = distance_mesuring_method(X, centroids)
    closest_cluster_ids = [row.index(min(row)) for row in distance_matrix]

    for i, cluster_id in enumerate(closest_cluster_ids):
        clusters[cluster_id].append(X[i])

    return clusters

# Function to check if centroids have covered the dataset
def has_centroids_covered(previous_centroids, new_centroids, distance_mesuring_method, movement_threshold_delta):
    distances = distance_mesuring_method(previous_centroids, new_centroids)
    max_movement = max(distances[i][i] for i in range(len(distances)))
    return max_movement <= movement_threshold_delta

# Function to perform k-means algorithm
def perform_k_means_algorithm(X, k, distance_mesuring_method, movement_threshold_delta):
    new_centroids = get_initial_centroids(X, k)
    centroids_covered = False

    while not centroids_covered:
        previous_centroids = new_centroids
        clusters = get_clusters(X, previous_centroids, distance_mesuring_method)

        new_centroids = [tuple(map(lambda x: sum(x) / len(x), zip(*clusters[key]))) if clusters[key] else previous_centroids[key]
                         for key in sorted(clusters.keys())]

        centroids_covered = has_centroids_covered(previous_centroids, new_centroids, distance_mesuring_method, movement_threshold_delta)

    return new_centroids

# Function to get the reduced colors image
def get_reduced_colors_image(image, number_of_colors):
    h, w, d = image.shape
    X = [tuple(image[i][j]) for i in range(h) for j in range(w)]

    centroids = perform_k_means_algorithm(X, number_of_colors, get_euclidean_distance, movement_threshold_delta=4)
    distance_matrix = get_euclidean_distance(X, centroids)
    closest_cluster_ids = [row.index(min(row)) for row in distance_matrix]

    X_reconstructed = [centroids[id] for id in closest_cluster_ids]
    reduced_image = [[X_reconstructed[i * w + j] for j in range(w)] for i in range(h)]

    return reduced_image

# Function to save the reduced colors image
def save_reduced_colors_image(reduced_image, filename):
    h, w = len(reduced_image), len(reduced_image[0])
    img_array = np.array(reduced_image).reshape(h, w, 3).astype(np.uint8)

    # Save the image
    cv2.imwrite(filename, img_array)

# Function to process and save the reduced colors image
def process_image(k, img, rank):
    reduced_image = get_reduced_colors_image(img, k)
    save_reduced_colors_image(reduced_image, f"image/k{k}_rank{rank}.webp")

# Main block
if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Root rank loads the image and initializes k-values
    if rank == 0:
        img = cv2.imread("image.webp")
        if img is None:
            print("Failed to load image.")
            img_shape = (0, 0, 0)
            img_data = np.array([], dtype=np.uint8)
        else:
            img_shape = img.shape
            img_data = img.flatten()
        k_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 16, 32, 64, 128]
        chunks = np.array_split(k_values, size)
    else:
        img_shape = None
        img_data = None
        chunks = None

    # Broadcast the shape and k-values
    img_shape = comm.bcast(img_shape, root=0)
    chunks = comm.scatter(chunks, root=0)

    # Non-root ranks initialize an empty array to receive the image data
    if rank != 0:
        img_data = np.empty(img_shape[0] * img_shape[1] * img_shape[2], dtype=np.uint8)

    # Broadcast the image data to all ranks
    comm.Bcast(img_data, root=0)

    # Reshape the received image data back into the original shape
    if rank != 0 and img_shape != (0, 0, 0):
        img = img_data.reshape(img_shape)

    # Start processing the image using each rank's chunk of k-values
    for k in chunks:
        start_time = time.time()
        process_image(k, img, rank)
        end_time = time.time()
        print(f"Rank {rank} - Time taken for k={k}: {end_time - start_time:.8f} seconds")
