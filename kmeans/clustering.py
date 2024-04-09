"""
@author: tjdwill 
@date: 5 April 2024
@title: K-means Clustering
@description:
    A more function-based implementation of my k-means clustering class.
"""
from typing import Union, List, Tuple, Callable
import numpy as np
from numpy.typing import NDArray
from kmeans.base_funcs import _assign_clusters, _validate, _new_centroids, dist_funcs, SMALLEST_THRESH


Clusterable = Union[List[NDArray], Tuple[NDArray], NDArray]
Clusters = dict[int: Clusterable]

class MaxIterationError(Exception):
    """An exception to be raised when the maximum iteration threshold is exceeded."""
    pass


def cluster(
        data: Union[List[NDArray], Tuple[NDArray], NDArray],
        k: int,*,
        initial_means: Union[List[NDArray], Tuple[NDArray], NDArray] = None,
        ndim: int = None,
        threshold: float = SMALLEST_THRESH, 
        max_iterations: int = 100,
        dist_func: Callable = "euclidean"
) -> tuple[Clusters, NDArray]:
    """Perform k-means clustering
    
    Args:
        data: The input data
            This data should be formatted in terms of row vectors.
            In other words, given a flat numpy array
            data=np.array([0, 1, 2, 3, 4]), do the following:
            `data = data.reshape(data.shape[-1], -1)`
            or `data = data[..., np.newaxis]`
            It should make each point a row entry:
                [[0], [1], [2], [3], [4]]
            Pass this version into the KMeans constructor.
            Data of higher dimensions (ex. a multi-channeled image)
            should be flattened using the number of indices
            for the deepest dimension. So, for an image with shape
            (480, 640, 3), run
                `data = data.reshape(-1, data.shape[-1])`
            and pass this data into the constructor.
        k: Amount of clusters
        initial_means: The initial cluster centroids
            Defaults to `None` -> Means are randomly selected from data 
            with uniform probability
        ndim: Dimension limit for clustering; 
            Defaults to None -> selects the ndim based on data dimensionality
        threshold: How much can a given cluster centroid 
            change between iterations. Default: 4.440892098500626e-15
        max_iterations: The counter timeout 
            Default: 100
        dist_func: Distance calculation method
            Default: 'euclidean'

    Returns:
        tuple[Clusterable, NDArray]: clustered data, cluster centroids

    Raises:
        MaxIterationError: Raise this exception if the clustering doesn't
            converge before reaching the `max_iterations` count.
    """
    data, initial_means, ndim = _validate(
        data,
        k, 
        initial_means=initial_means, 
        ndim=ndim,
        threshold=threshold, 
        max_iterations=max_iterations
    )

    clusters, old_centroids = {}, initial_means

    for _ in range(max_iterations):
        clusters = _assign_clusters(data, old_centroids, dist_funcs[dist_func])
        centroids = _new_centroids(clusters, ndim)
        changes = np.linalg.norm(centroids - old_centroids, axis=1)  # Distance along each vector
        if any(np.where(changes > threshold, True, False)):
            old_centroids = centroids
        else:
            return clusters, centroids
    else:
        raise MaxIterationError("Iteration count exceeded.")            
