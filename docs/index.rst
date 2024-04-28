.. K-Means Clustering documentation master file, created by
   sphinx-quickstart on Sat Apr 27 13:15:22 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to K-Means Clustering's documentation!
==============================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   modules

This is a package for easily performing k-means clustering on a set of data. 
Unique to this implementation is its ability to specify the dimension up to
which the data is clustered. For example::

   >>> from kmeans import cluster
   >>> import numpy as np
   >>> np.random.seed(27)   # For reproducible results
   >>> data = np.random.random((15, 5)).round(3)
   >>> data[0]
   array([0.426, 0.815, 0.735, 0.868, 0.383])
   >>> # Cluster using only first two dimensions
   >>> clusters, centroids = cluster(data, k=3, ndim=2, tolerance=0.001)
   >>> centroids
   array([[0.9004, 0.79  ],
       [0.5795, 0.3995],
       [0.254 , 0.6325]])
   >>> clusters  # visually compare centroids with first two elements of each data entry.
   {0: array([[0.979, 0.893, 0.21 , 0.742, 0.663],
        [0.887, 0.858, 0.749, 0.87 , 0.187],
        [0.966, 0.583, 0.092, 0.014, 0.837],
        [0.915, 0.705, 0.387, 0.706, 0.923],
        [0.755, 0.911, 0.242, 0.976, 0.304]]),
    1: array([[0.326, 0.373, 0.794, 0.151, 0.17 ],
        [0.701, 0.181, 0.599, 0.415, 0.514],
        [0.477, 0.493, 0.595, 0.076, 0.117],
        [0.489, 0.596, 0.264, 0.992, 0.21 ],
        [0.583, 0.649, 0.911, 0.122, 0.676],
        [0.901, 0.105, 0.673, 0.87 , 0.561]]),
    2: array([[0.426, 0.815, 0.735, 0.868, 0.383],
        [0.081, 0.305, 0.783, 0.163, 0.071],
        [0.221, 0.726, 0.849, 0.929, 0.736],
        [0.288, 0.684, 0.52 , 0.877, 0.924]])}

Installation
============

The package can be installed via `PyPI <https://pypi.org/project/kmeans-tjdwill/>`_:

.. code-block:: console

   $ pip install kmeans-tjdwill

Issues
======

Create an issue on the `GitHub page <https://github.com/tjdwill/kmeans/issues>`_.
Please be civil, professional, and kind.

License
=======

This work is licensed under the `MIT License <https://github.com/tjdwill/kmeans/blob/main/LICENSE>`_.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
