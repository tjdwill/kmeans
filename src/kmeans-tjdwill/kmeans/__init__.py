# Set version num
import importlib.metadata
__version__ = importlib.metadata.version("kmeans-tjdwill")

# Set user API
from kmeans.clustering import cluster, MaxIterationError
from kmeans.animate import view_clustering
__all__ = ["cluster", "view_clustering", "MaxIterationError"]