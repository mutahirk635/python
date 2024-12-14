import random
class K_Mean:
    def __init__(self, n_cluster=2, max_iteration=100):
        self.n_cluster = n_cluster
        self.max_iteration = max_iteration
        self.centroid = None

    def Predict(self, X):
        random_index=random.sample(range(0,X.shape[0]),self.n_cluster)
        self.centroid=X[random_index]

    for i in range(self,X):





