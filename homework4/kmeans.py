import json
import random
import numpy as np


def cluster_points(X, mu):
    """
    Distribute data points into clusters.

    Inputs:
    - X: A list of data points in 2d space, each elements is a list of 2
    - mu: A list of K cluster centers, each elements is a list of 2

    Returns:
    - clusters: A dict, keys are cluster index {1,2, ..., K} (int),
                value are the list of corresponding data points.
    """

    clusters = {}

    # you need to fill in your solution here
    num_clusters = len(mu)

    clusters_list = [[] for i in range(num_clusters)]

    for x in X:

        p1 = np.array(x)

        minimizer = -1
        min_distance = float("inf")

        for i_cluster in range(num_clusters):

            p2 = np.array(mu[i_cluster])

            dist = np.linalg.norm(p1-p2)

            if dist < min_distance:

                minimizer = i_cluster
                min_distance = dist

        clusters_list[minimizer].append(x)

    for index in range(1, num_clusters+1):
        clusters[index] = clusters_list[index-1]


    return clusters


def reevaluate_centers(mu, clusters):
    """
    Update cluster centers.

    Inputs:
    - mu: A list of K cluster centers, each elements is a list of 2
    - clusters: A dict, keys are cluster index {1,2, ..., K} (int),
                value are the list of corresponding data points.

    Returns:
    - newmu: A list of K updated cluster centers, each elements is a list of 2
    """
    newmu = []

    # you need to fill in your solution here

    num_clusters = len(mu)

    for index in range(num_clusters):

        data_points = clusters[index+1]

        center_sum = [0, 0]

        if len(data_points) != 0:

            for i in range(len(data_points)):

                center_sum = [center_sum[0] + data_points[i][0],center_sum[1] + data_points[i][1]]

            center = [center_sum[j]/len(data_points) for j in range(2)]

            newmu.append(center)

        else:
            newmu.append(mu[index])

    return newmu


def has_converged(mu, oldmu):
    return set([tuple(a) for a in mu]) == set([tuple(a) for a in oldmu])


def find_centers(X, K):
    # Initialize to K random centers
    random.seed(100)
    oldmu = random.sample(X, K)
    mu = random.sample(X, K)
    while not has_converged(mu, oldmu):
        oldmu = mu
        # Assign all points in X to clusters
        clusters = cluster_points(X, mu)
        # Reevaluate centers
        mu = reevaluate_centers(oldmu, clusters)

    return(mu, clusters)


def kmeans_clustering():
    # load data
    with open('hw4_circle.json', 'r') as f:
        data_circle = json.load(f)
    with open('hw4_blob.json', 'r') as f:
        data_blob = json.load(f)

    mu_all = {}
    clusters_all = {}
    for K in [2, 3, 5]:
        key = 'blob_K=' + str(K)
        mu_all[key], clusters_all[key] = find_centers(data_blob, K)
        key = 'circle_K=' + str(K)
        mu_all[key], clusters_all[key] = find_centers(data_circle, K)

    return mu_all, clusters_all


def main():
    mu_all, clusters_all = kmeans_clustering()

    print('K-means Cluster Centers:')
    for key, value in mu_all.items():
        print('\n%s:'% key)
        print(np.array_str(np.array(value), precision=4))

    with open('kmeans.json', 'w') as f_json:
        json.dump({'mu': mu_all, 'clusters': clusters_all}, f_json)


if __name__ == "__main__":
    main()