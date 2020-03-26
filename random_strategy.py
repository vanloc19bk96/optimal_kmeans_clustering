from initial_centers_strategies import Initial_Centers_Strategies
import numpy as np

class Random_Strategy(Initial_Centers_Strategies):
    def __init__(self, X, n_clusters=8, random_state = None):
        super().__init__()
        self.X = X
        self.n_clusters = n_clusters
        np.random.seed(random_state)

    def create_centers(self):
        indicates = np.random.randint(0, len(self.X), size=self.n_clusters)
        initial_centers = self.X[indicates]
        return initial_centers

