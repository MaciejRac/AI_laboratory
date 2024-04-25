import numpy as np

def initialize_centroids_forgy(data, k):
    # TODO implement random initialization
    i = np.random.choice(data.shape[0], k, replace=False)
    centroids=data[i]
    print(type(centroids))
    return centroids

def initialize_centroids_kmeans_pp(data, k):
    # TODO implement kmeans++ initizalization
    centroids=[]
    sum_distances:int=0
    i=(np.random.choice(data.shape[0]))
    centroids.append(data[i])
    for _ in range(k-1):
        distances = np.array([min(np.linalg.norm(point - centroid) for centroid in centroids) for point in data])
        # for dist in distances:
        #     sum_distances+=dist
        # probabilities = distances / distances.sum()
        # new_centroid = data[np.random.choice(range(data.shape[0]), p=probabilities)]
        # centroids.append(new_centroid)
        new_centroid = data[np.argmax(distances)]
        centroids.append(new_centroid)
    return np.array(centroids)

def assign_to_cluster(data, centroids):
    # TODO find the closest cluster for each data point
    cluster_assignments = []
    for point in data:
        distances = [np.linalg.norm(point - centroid) for centroid in centroids]
        closest_cluster = np.argmin(distances)
        cluster_assignments.append(closest_cluster)
    
    return np.array(cluster_assignments)

def update_centroids(data, assignments):
    # TODO find new centroids based on the assignments
    num_centroids = len(np.unique(assignments))
    centroids = np.zeros((num_centroids, data.shape[1]))
    
    for i in range(num_centroids):
        points_for_centroid = data[np.array(assignments) == i]
        new_centroid = np.mean(points_for_centroid, axis=0)
        centroids[i] = new_centroid
    
    return centroids

def mean_intra_distance(data, assignments, centroids):
    return np.sqrt(np.sum((data - centroids[assignments, :])**2))

def k_means(data, num_centroids, kmeansplusplus= False):
    # centroids initizalization
    if kmeansplusplus:
        centroids = initialize_centroids_kmeans_pp(data, num_centroids)
    else: 
        centroids = initialize_centroids_forgy(data, num_centroids)

    
    assignments  = assign_to_cluster(data, centroids)
    for i in range(100): # max number of iteration = 100
        print(f"Intra distance after {i} iterations: {mean_intra_distance(data, assignments, centroids)}")
        centroids = update_centroids(data, assignments)
        new_assignments = assign_to_cluster(data, centroids)
        if np.all(new_assignments == assignments): # stop if nothing changed
            break
        else:
            assignments = new_assignments

    return new_assignments, centroids, mean_intra_distance(data, new_assignments, centroids)         

