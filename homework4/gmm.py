import json
import random
import numpy as np


def gmm_clustering(X, K):
    """
    Train GMM with EM for clustering.

    Inputs:
    - X: A list of data points in 2d space, each elements is a list of 2
    - K: A int, the number of total cluster centers

    Returns:
    - mu: A list of all K means in GMM, each elements is a list of 2
    - cov: A list of all K covariance in GMM, each elements is a list of 4
            (note that covariance matrix is symmetric)
    """

    # Initialization:
    pi = []
    mu = []
    cov = []
    for k in range(K):
        pi.append(1.0 / K)
        mu.append(list(np.random.normal(0, 0.5, 2)))
        temp_cov = np.random.normal(0, 0.5, (2, 2))
        temp_cov = np.matmul(temp_cov, np.transpose(temp_cov))
        cov.append(list(temp_cov.reshape(4)))

    ### you need to fill in your solution starting here ###

    # Run 100 iterations of EM updates
    for t in range(100):

        # Calculate gamma's
        gamma = np.zeros((len(X),K))

        for x_index in range(len(X)):
            gamma_temp = []
            for k in range(K):

                x_array = np.array(X[x_index]).reshape(2,1)
                mu_array = np.array(mu[k]).reshape(2,1)
                cov_array = np.array(cov[k]).reshape(2,2)

                Normal_dist = (1.0/np.sqrt(np.linalg.det(cov_array)))*np.exp(-0.5*np.dot(np.transpose(x_array - mu_array),
                                                                             np.dot(np.linalg.inv(cov_array),(x_array - mu_array))))

                gamma_temp.append(pi[k]*Normal_dist[0][0])

            gamma_temp_array = np.array(gamma_temp)
            gamma_temp_array = gamma_temp_array/np.sum(gamma_temp_array)

            gamma[x_index,:] = gamma_temp_array


        # Update pi,mu,cov
        pi = []
        mu = []
        cov = []

        sum_gammas = np.sum(gamma)

        for k in range(K):
            pi.append(np.sum(gamma[:,k])/sum_gammas)

            X_array = np.array(X)

            mean_temp = np.dot(X_array.transpose(),gamma[:,k])/np.sum(gamma[:,k])

            mu.append(list(mean_temp))

            mu_repmat = np.tile(mu[k],[len(X),1])

            X_array_centered = X_array - mu_repmat

            temp_cov = np.dot(np.multiply(gamma[:,k],X_array_centered.transpose()),X_array_centered)/np.sum(gamma[:,k])

            cov.append(list(temp_cov.reshape(4)))


    return mu, cov


def main():
    # load data
    with open('hw4_blob.json', 'r') as f:
        data_blob = json.load(f)

    mu_all = {}
    cov_all = {}

    print('GMM clustering')
    for i in range(5):
        np.random.seed(i)
        mu, cov = gmm_clustering(data_blob, K=3)
        mu_all[i] = mu
        cov_all[i] = cov

        print('\nrun' + str(i) + ':')
        print('mean')
        print(np.array_str(np.array(mu), precision=4))
        print('\ncov')
        print(np.array_str(np.array(cov), precision=4))

    with open('gmm.json', 'w') as f_json:
        json.dump({'mu': mu_all, 'cov': cov_all}, f_json)


if __name__ == "__main__":
    main()