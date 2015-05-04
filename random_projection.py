from sklearn.random_projection import SparseRandomProjection, GaussianRandomProjection
import numpy as np
from scipy.spatial.distance import euclidean
import data_factory as df
import time
import random
from sklearn.preprocessing import OneHotEncoder

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import os
import sys

'''
    compare different implementations of random projection
    usage:

    dataset is:
    - https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/

    $ ptyhon random_projection.py iterations 10


'''

if len(sys.argv) != 2:
    print "only / max one param allowed"
    exit()

params = sys.argv

id = params[1]

if not all.has_key(id):
    print "experiment with id '%s' not found" % id
    exit()

run.execute(all[id])



data, label, desc, size = df.loadFirstCancerDataset()

enc = OneHotEncoder()
enc.fit(data)
dataReduced = enc.transform(data).toarray()

print np.shape(data)
print np.shape(dataReduced)

'''
# amount of rows
orig_rows = len(data)
origDistances = np.empty((orig_rows, orig_rows))

items = range(len(data))
for i in items:
    for j in items:
        if i == j:
            origDistances[i][j] = .0
        else:
            origDistances[i][j] = euclidean(data[i], data[j])

if hasattr(data, "toarray"):
    data = data.toarray()

orig_dimension = len(data[0])

# compare original data with reduced data
def measureDistances(data, reduced, desc):

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
    meanDistance = np.abs(np.mean(origDistances - newDistancs))
    return meanDistance

# measure duration and run the reduction
def reduceAndMeasure(action, data):
    start = time.time()
    reduced = action(data)
    duration = time.time() - start
    return reduced, duration

# scikit-learn implementation: gaussian matrix
def gaussianRP(data, new_dimension):
    rp = GaussianRandomProjection(n_components=new_dimension)
    return rp.fit_transform(data)

# scikit-learn implementation: sparse matrix
def sparseRP(data, new_dimension):
    rp = SparseRandomProjection(n_components=new_dimension)
    return rp.fit_transform(data)

# just extract the random matrix from the api
def otherScikitImpl(data, new_dimension):
    rp = GaussianRandomProjection(n_components=new_dimension)
    m = rp._make_random_matrix(new_dimension, orig_dimension)
    m = np.mat(m)
    reduced = m * np.mat(data).transpose()
    reduced = reduced.transpose()
    return reduced

random = np.random.mtrand._rand

# naive implementation of the random matrix
def custom1(data, new_dimension):
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
def custom2(data, new_dimension):
    m = np.empty((orig_dimension, new_dimension))
    for i in range(len(m)):
        for j in range(len(m[i])):
            m[i][j] = random.random()

    reduced = np.mat(data) * m
    return reduced

actions = {
    "gaussian RP": gaussianRP,
    "sparse RP": sparseRP,
    "other scikit": otherScikitImpl,
    "custom 1": custom1,
    "custom 2": custom2
}



results = dict()

# 10 iterations
x = np.arange(1,10)
for key in actions.iterkeys():
    print key
    durations = list()
    distances = list()
    di = list()
    du = list()
    for i in x:
        print " %s" % i
        action = actions[key]

        reduced, d = reduceAndMeasure(action, data)
        dist = measureDistances(data, reduced, key)
        du.append(d)
        di.append(dist)

        durations.append(np.mean(du))
        distances.append(np.mean(di))

    results[key] = dict()
    results[key]["durations"] = durations
    results[key]["distances"] = distances


plt.subplot(211)
plt.grid()
plt.xlabel("iterations")
plt.ylabel("mean distance")

for key in results.iterkeys():
    plt.plot(x, results[key]["distances"], label=key)

plt.legend(loc="best")
#plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08), fancybox=True, shadow=True, ncol=2)

plt.subplot(212)
plt.grid()
plt.xlabel("iterations")
plt.ylabel("mean duration")

for key in results.iterkeys():
    plt.plot(x, results[key]["durations"], label=key)

plt.legend(loc="best")
#plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08), fancybox=True, shadow=True, ncol=2)

outputFolder = os.path.dirname(os.path.abspath(__file__))
outputFolder = "%s/output" % outputFolder

plt.savefig( "%s/rp_1.png" % outputFolder, dpi=320, bbox_inches = "tight")
'''