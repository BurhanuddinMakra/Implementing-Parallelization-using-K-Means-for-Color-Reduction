import random
import cv2
import time

random.seed(7)


def get_initial_centroids(X, k):
    number_of_samples = len(X)
    sample_points_ids = random.sample(range(0, number_of_samples), k)

    centroids = [tuple(X[id]) for id in sample_points_ids]
    unique_centroids = list(set(centroids))

    number_of_unique_centroids = len(unique_centroids)

    while number_of_unique_centroids < k:
        new_sample_points_ids = random.sample(range(0, number_of_samples), k - number_of_unique_centroids)
        new_centroids = [tuple(X[id]) for id in new_sample_points_ids]
        unique_centroids = list(set(unique_centroids + new_centroids))

        number_of_unique_centroids = len(unique_centroids)

    return unique_centroids


def get_euclidean_distance(A, B):
    distances = []
    for a in A:
        row = []
        for b in B:
            # Ensure no overflow by controlling the data type
            distance = sum((int(a_i) - int(b_i)) ** 2 for a_i, b_i in zip(a, b)) ** 0.5
            row.append(distance)
        distances.append(row)
    return distances

def get_clusters(X, centroids, distance_mesuring_method):
    k = len(centroids)

    clusters = {i: [] for i in range(k)}

    distance_matrix = distance_mesuring_method(X, centroids)
    closest_cluster_ids = [min(range(k), key=lambda i: distance_matrix[j][i]) for j in range(len(X))]

    for i, cluster_id in enumerate(closest_cluster_ids):
        clusters[cluster_id].append(X[i])

    return clusters


def has_centroids_covered(previous_centroids, new_centroids, distance_mesuring_method, movement_threshold_delta):
    distances_between_old_and_new_centroids = distance_mesuring_method(previous_centroids, new_centroids)
    centroids_covered = max(distances_between_old_and_new_centroids[i][i] for i in range(len(previous_centroids))) <= movement_threshold_delta

    return centroids_covered


def perform_k_means_algorithm(X, k, distance_mesuring_method, movement_threshold_delta=4):
    new_centroids = get_initial_centroids(X=X, k=k)

    centroids_covered = False
    iterations = 0

    while not centroids_covered:
        iterations += 1
        previous_centroids = new_centroids
        clusters = get_clusters(X, previous_centroids, distance_mesuring_method)

        new_centroids = [tuple(sum(point[i] for point in clusters[key]) / len(clusters[key]) for i in range(len(X[0])))
                         for key in sorted(clusters.keys())]

        centroids_covered = has_centroids_covered(previous_centroids, new_centroids, distance_mesuring_method, movement_threshold_delta)
    print(iterations)
    print(new_centroids)

    return new_centroids


def get_reduced_colors_image(image, number_of_colors):
    h, w, d = image.shape

    X = [tuple(image[i, j]) for i in range(h) for j in range(w)]

    centroids = perform_k_means_algorithm(X, k=number_of_colors, distance_mesuring_method=get_euclidean_distance, movement_threshold_delta=4)
    distance_matrix = get_euclidean_distance(X, centroids)
    closest_cluster_ids = [min(range(len(centroids)), key=lambda i: distance_matrix[j][i]) for j in range(len(X))]

    X_reconstructed = [centroids[closest_cluster_ids[i]] for i in range(len(X))]
    X_reconstructed = [(int(r), int(g), int(b)) for r, g, b in X_reconstructed]  # Ensure the types are correct for OpenCV
    reduced_image = np.array(X_reconstructed, dtype=np.uint8).reshape((h, w, d))  # Correct reshaping

    return reduced_image

if __name__ == '__main__':
    start_time = time.time()

    k_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 16, 32, 64, 128]

    img = cv2.imread("image.webp")

    for k in k_values:
        start_function_time = time.time()
        reduced_colors_image = get_reduced_colors_image(img, k)
        end_function_time = time.time()
        function_time = end_function_time - start_function_time
        print(f"Time taken for get_reduced_colors_image with k={k}: {function_time:.8f} seconds")

        cv2.imwrite(f"image/k{k}.webp", reduced_colors_image)

    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total time taken: {total_time:.8f} seconds")

