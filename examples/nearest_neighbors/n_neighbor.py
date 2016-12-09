from pandas.core.nanops import disallow
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.neighbors import NearestNeighbors
import numpy as np
from sympy.core.basic import preorder_traversal


X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
y = np.array([1, 2, 3, 4, 5, 6])
clf = NearestCentroid()
clf.fit(X, y)
print(clf.predict([[5, 3]]))

#nbrs = NearestNeighbors(n_neighbors=3, algorithm='auto').fit(test, y)

#distances, indices = nbrs.kneighbors(test)

#print(distances)
#print(indices)

