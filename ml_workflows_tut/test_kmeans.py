# -*- coding: utf-8 -*-

import dataset
import kmeans

# Load dataset
iris_data = dataset.load_dataset('iris.csv')

# Convert class names to numeric representations
iris_data, iris_classes = dataset.to_numeric(iris_data, 'species')

# Convert dataframe strings to floats
attrs_conv = list(iris_data.axes[1][:-1])
data = dataset.from_str(iris_data, attrs_conv)

# Covert dataset to matrix representation
iris_ds = dataset.to_matrix(iris_data)

# Perform k-means clustering
centroids, cluster_assignments, iters, orig_centroids = kmeans.cluster(iris_ds, 3)

# Output results
print('Number of iteratios: ', iters)
print('\nFinal Centroids: ', centroids)
print('\nCluster membership and error of first 10 instances:\n ', cluster_assignments[:10])
print('\nOriginal centroids:\n', orig_centroids)