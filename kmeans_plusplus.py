from initial_centers_strategies import Initial_Centers_Strategies
import numpy as np
from scipy.spatial import distance
from numpy.linalg import norm


class KMeansPlusPlus(Initial_Centers_Strategies):
    def __init__(self, X, n_clusters=3, random_state = None):
        super().__init__()
        self.X = X
        self.n_clusters = n_clusters
        np.random.seed(random_state)

    def remove_center_out_of_X(self, value):
        for index, x in enumerate(self.X):
            if np.array_equal(x, value):
                self.X = np.delete(self.X, index, axis=0)

    def create_centers(self):
        self.X = np.unique(self.X, axis=0)

        error_message = "Number of distict clusters ({}) found smaller than n_clusters ({}). " \
                                               "Possibly due to duplicate points in X".format(len(self.X), self.n_clusters)
        assert self.n_clusters <= len(self.X),error_message
        indice = np.random.randint(0, len(self.X), size=1)
        centers = {}

        # find the first center
        first_center = self.X[indice, :]
        first_center = first_center.squeeze(0)
        if self.n_clusters == 1:
            return np.asarray(first_center)
        self.remove_center_out_of_X(first_center)
        # find the second center
        distances = [distance.euclidean(x, first_center) for x in self.X]
        furthest = np.argmax(distances)
        second_center = self.X[furthest, :]
        self.remove_center_out_of_X(second_center)
        count_clusters = 2
        centers[0] = first_center
        centers[1] = second_center

        while count_clusters < self.n_clusters:
            nearest_each_center = {key: [] for key in range(count_clusters)}

            # find features which are nearest center
            for x in self.X:
                distances = [distance.euclidean(x, centers[center]) for center in centers]
                nearest_each_center[np.argmin(distances)].append(x)

            # find the new center
            max_distance = 0
            best_center = 0
            for center, features in nearest_each_center.items():
                if len(features) != 0:
                    max_distance_each_cluster = np.max(norm(centers[center] - features, axis=1))
                    indice = np.argmax(norm(centers[center] - features, axis=1))
                    if max_distance < max_distance_each_cluster:
                        max_distance = max_distance_each_cluster
                        best_center = features[indice]
            centers[count_clusters] = best_center
            nearest_each_center[count_clusters] = []
            count_clusters += 1
            self.remove_center_out_of_X(best_center)
        return np.asarray([value for key, value in centers.items()])