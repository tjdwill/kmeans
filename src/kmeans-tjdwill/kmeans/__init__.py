# Set version num
__version__ = "1.0.3"

# Set user API
from kmeans.clustering import cluster, MaxIterationError
from kmeans.animate import view_clustering
__all__ = ["cluster", "view_clustering", "MaxIterationError"]