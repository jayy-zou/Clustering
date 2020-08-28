import numpy as np
from sklearn.datasets import make_blobs

def generate_cluster_data(
    n_samples=100,
    n_features=2,
    n_centers=2,
    cluster_stds=1.0
):
   
    return make_blobs(
        n_samples=n_samples, n_features=n_features, centers=n_centers, cluster_std=cluster_stds
    )
