from random_strategy import Random_Strategy
import numpy as np
from kmeans_plusplus import KMeansPlusPlus
import matplotlib.pyplot as plt
import copy

class Kmeans(object):
    def __init__(self, n_clusters=3, strategy='random', max_iter=500, epsilon=1e-4, random_state=42):
        self.n_clusters = n_clusters
        self.strategy = strategy
        self.max_iter = max_iter
        self.epsilon = epsilon
        self.random_state = random_state

    def fit(self, X):
        centers = {}
        if self.strategy == 'random':
            random_strategy = Random_Strategy(X, self.n_clusters, self.random_state)
            centers = random_strategy.create_centers()
        elif self.strategy == 'kmeans++':
            kmeans_plusplus = KMeansPlusPlus(X, self.n_clusters, self.random_state)
            centers = kmeans_plusplus.create_centers()
        if self.n_clusters == 1:
            return {0: centers}
        centers = {i : value for i, value in enumerate(centers)}
        features = {key: [] for key in range(self.n_clusters)}
        iter = 0

        while iter < self.max_iter:
            for x in X:
                distances = [np.linalg.norm(x - centers[center]) for center in centers]
                nearest = np.argmin(distances)
                features[nearest].append(x)
            old_centers = copy.deepcopy(centers)
            is_optimal = True
            for feature in features:
                centers[feature] = np.average(features[feature], axis=0)
            for center in centers:
                if sum((centers[center] - old_centers[center]) / old_centers[center]) * 100 > self.epsilon:
                    is_optimal = False
            iter += 1
            if is_optimal:
                break
        print("Number of iteration", iter)
        return centers

if __name__ == '__main__':
    kmeans = Kmeans(strategy='kmeans++')
    means = [[2, 2], [8, 3], [3, 6]]
    cov = [[1, 0], [0, 1]]
    N = 20
    X0 = np.random.multivariate_normal(means[0], cov, N)
    X1 = np.random.multivariate_normal(means[1], cov, N)
    X2 = np.random.multivariate_normal(means[2], cov, N)
    original_label = np.asarray([0] * N + [1] * N + [2] * N).T
    X = np.concatenate((X0, X1, X2), axis=0)
    K = 3
    centers = kmeans.fit(X)
    X0 = X[original_label == 0, :]
    X1 = X[original_label == 1, :]
    X2 = X[original_label == 2, :]

    plt.plot(X0[:, 0], X0[:, 1], 'b^', markersize=4, alpha=.8)
    plt.plot(X1[:, 0], X1[:, 1], 'go', markersize=4, alpha=.8)
    plt.plot(X2[:, 0], X2[:, 1], 'rs', markersize=4, alpha=.8)
    plt.plot(centers[0][0], centers[0][1], 'y^', markersize=20, alpha=.8)
    plt.plot(centers[1][0], centers[1][1], 'y^', markersize=20, alpha=.8)
    plt.plot(centers[2][0], centers[2][1], 'y^', markersize=20, alpha=.8)
    plt.show()

