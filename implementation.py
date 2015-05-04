from sklearn.random_projection import SparseRandomProjection, GaussianRandomProjection
import numpy as np
import time
from scipy.spatial.distance import euclidean
import random

# measure duration and run the reduction
def reduceAndMeasure(action, data, orig_dimension, new_dimension):
    start = time.time()
    reduced = action(data, orig_dimension, new_dimension)
    duration = time.time() - start
    return reduced, duration

# scikit-learn implementation: gaussian matrix
def gaussianRP(data,orig_dimension, new_dimension):
    rp = GaussianRandomProjection(n_components=new_dimension)
    return rp.fit_transform(data)

# scikit-learn implementation: sparse matrix
def sparseRP(data, orig_dimension, new_dimension):
    rp = SparseRandomProjection(n_components=new_dimension)
    return rp.fit_transform(data)

# just extract the random matrix from the api
def otherScikitImpl(data,orig_dimension, new_dimension):
    rp = GaussianRandomProjection(n_components=new_dimension)
    m = rp._make_random_matrix(new_dimension, orig_dimension)
    m = np.mat(m)
    reduced = m * np.mat(data).transpose()
    reduced = reduced.transpose()
    return reduced

# random = np.random.mtrand._rand

# naive implementation of the random matrix
def custom1(data, orig_dimension, new_dimension):
    minusOne = 0.1
    one = 0.9
    rows = len(data)
    m = np.empty((orig_dimension, new_dimension))
    # build random matrix
    for i in range(len(m)):
        for j in range(len(m[i])):
            rand = random.random()
            if rand < minusOne:
                m[i][j] = -1
            elif rand >= one:
                m[i][j] = 1
            else:
                m[i][j] = 0

    reduced = np.mat(data) * m
    return reduced

# non-sense implementation for comparison
def custom2(data, orig_dimension, new_dimension):
    m = np.empty((orig_dimension, new_dimension))
    for i in range(len(m)):
        for j in range(len(m[i])):
            m[i][j] = random.random()

    reduced = np.mat(data) * m
    return reduced




actions = {
    "gaussian RP": gaussianRP,
    "sparse RP": sparseRP,
    "manual scikit": otherScikitImpl,
    "custom 1": custom1,
    "custom 2": custom2
}


# compare original data with reduced data
def measureDistances(origDistances, data, reduced, desc):

    a = np.shape(data)
    b = np.shape(reduced)
    if a[0] != b[0]:
        raise Exception("%s: same amount of instances required. data: %s, reduced: %s" % (desc, a,b))

    newDistancs = np.empty((b[0], b[0]))
    items = range(b[0])
    for i in items:
        for j in items:
            if i == j:
                newDistancs[i][j] = 0
            else:
            #if i % 5 == 0 and j % 10 == 0:
                newDistancs[i][j] = euclidean(reduced[i], reduced[j])

    # compare item by item
    meanDistance = np.abs(np.mean( - newDistancs))
    return meanDistance

