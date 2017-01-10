import numpy as np
from sklearn.semi_supervised import label_propagation

import numpy as np



class Hans:

    dim = 10
    max_random = 3.

    X = np.empty((dim,3))
    y = np.empty(dim)
    value = -1

    for i in range(dim):

        if i % 5 == 0:
            value += 1

    X[0] = np.asarray([0., 0., 1], dtype=np.float64)
    X[1] = np.asarray([0., 1., 1], dtype=np.float64)
    X[2] = np.asarray([0., 2., 1], dtype=np.float64)
    X[3] = np.asarray([1., 0., 1], dtype=np.float64)
    X[4] = np.asarray([1., 1., 1], dtype=np.float64)
    X[5] = np.asarray([1., 2., 1], dtype=np.float64)
    X[6] = np.asarray([2., 0., 1], dtype=np.float64)
    X[7] = np.asarray([2., 1., 1], dtype=np.float64)
    X[8] = np.asarray([2., 2., 1], dtype=np.float64)
    X[9] = np.asarray([3, 3, 1], dtype=np.float64)

    y[0] = np.asarray([0], dtype=np.int)
    y[1] = np.asarray([-1], dtype=np.int)
    y[2] = np.asarray([-1], dtype=np.int)
    y[3] = np.asarray([1], dtype=np.int)
    y[4] = np.asarray([-1], dtype=np.int)
    y[5] = np.asarray([-1], dtype=np.int)
    y[6] = np.asarray([2], dtype=np.int)
    y[7] = np.asarray([-1], dtype=np.int)
    y[8] = np.asarray([-1], dtype=np.int)
    y[9] = np.asarray([3], dtype=np.int)

    print(X)
    print(y)

    print(X.shape)
    print(y.shape)

    clf = label_propagation.LabelPropagation()
    clf.fit(X, y)
    print(clf.predict(X))


