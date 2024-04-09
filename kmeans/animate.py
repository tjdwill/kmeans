
"""
@author: tjdwill 
@date: 5 April 2024
@title: K-means Clustering Animation
@description:
    Animating k-means for 2-D and 3-D cases.
"""
from typing import Union, List, Tuple, Callable
import numpy as np
from numpy.typing import NDArray
from matplotlib import pyplot as plt
from matplotlib import colors
import matplotlib as mpl
from kmeans.base_funcs import _assign_clusters, _validate, _new_centroids, dist_funcs, SMALLEST_THRESH


Clusterable = Union[List[NDArray], Tuple[NDArray], NDArray]
Clusters = dict[int: Clusterable]

class MaxIterationError(Exception):
    """An exception to be raised when the maximum iteration threshold is exceeded."""
    pass



colors: list = [color for color in colors.TABLEAU_COLORS.values()]
WRAP_FACTOR: int = len(colors)
SZ = 10
CENTROID_SZ = 2*SZ
def draw(clusters:Clusters, centroids: NDArray, ax_obj: mpl.axes.Axes, ndim: int) -> None:
    """Draws the clusters onto the figure
    
    Args:
        clusters: The segmented data
        centroids: The centers of the clusters
        ax: The axes object (from the figure)
        ndim: The number of dimensions

    Returns:
        None
    """
    k = len(clusters)
    ax_obj.clear()
    # Get the data
    labels = ['Cluster {}'.format(i) for i in range(k)]

    ax_obj.grid(which="both")
    ax_obj.set(xlabel="X", ylabel="Y", title=f"$k$ = ${k}$")

    for key in clusters:
        cluster = clusters[key]
        if ndim == 2:
            data = [(arr[0], arr[1]) for arr in cluster]
            x, y = list(zip(*data))
            cx, cy = centroids[key]

            ax_obj.scatter(
                x, y,
                s=SZ, c=colors[key%WRAP_FACTOR],
                label=labels[key], zorder=3,
            )
            ax_obj.scatter(
                cx, cy,
                s=CENTROID_SZ, c=colors[key%WRAP_FACTOR], edgecolors='k',
                zorder=3,
            ) 
        else:
            data = [(arr[0], arr[1], arr[2]) for arr in cluster]
            x, y, z = list(zip(*data))
            cx, cy, cz = centroids[key]
            ax_obj.scatter(
                x, y, z,
                s=SZ, c=colors[key%WRAP_FACTOR],
                label=labels[key], zorder=3,
            )
            ax_obj.scatter(
                cx, cy, cz,
                s=CENTROID_SZ, c=colors[key%WRAP_FACTOR], edgecolors='k',
                zorder=5,
            )
    else:
        ax_obj.legend()
        plt.pause(0.1) 
        plt.show(block=False)
        return


def view_clustering(
        data: Union[List[NDArray], Tuple[NDArray], NDArray],
        k: int,*,
        initial_means: Union[List[NDArray], Tuple[NDArray], NDArray] = None,
        ndim: int = None,
        threshold: float = SMALLEST_THRESH,
        max_iterations: int = 100,
        dist_func: Callable = "euclidean"
) -> tuple[Clusters, NDArray, mpl.figure.Figure]:
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
        tuple[Clusterable, NDArray, mpl.figure.Figure]:
            clustered data, cluster centroids, Matplotlib Figure

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
    print(ndim)
    # 2D or 3D plots?
    if ndim==2:
        fig, ax = plt.subplots()
    elif ndim == 3:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
    else:
        raise ValueError(
            "Only 2-D or 3-D may be animated. "
            "Use the `cluster` function for other dimensioned data."
        )

    for _ in range(max_iterations):
        clusters = _assign_clusters(data, old_centroids, dist_funcs[dist_func])
        centroids = _new_centroids(clusters, ndim)
        draw(clusters, centroids, ax, ndim)
        changes = np.linalg.norm(centroids - old_centroids, axis=1)  # Distance along each vector
        if any(np.where(changes > threshold, True, False)):
            old_centroids = centroids
        else:
            ax = fig.get_axes()[0]
            ax.set(title=f"Clustering @ $k$=${k}$")
            return clusters, centroids, fig
    else:
        raise MaxIterationError("Iteration count exceeded.")            
