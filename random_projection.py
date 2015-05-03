from sklearn.random_projection import SparseRandomProjection, GaussianRandomProjection
import numpy as np
from scipy.spatial.distance import euclidean
import data_factory as df
import time
import random

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import os

new_dimension = 10

#data = [[2,30,2,10,2,3,1,2,3,21,3,12,3,3,12,3,123,213,213,2],
#        [25,3,20,1,20,35,1,2,3,21,3,172,3,3,12,3,1723,213,213,72]]

#data, label, desc, size = df.loadFifthPlistaDataset()
data, label, desc, size = df.loadFirstCancerDataset()

orig_rows = len(data)

# calc n*n matrix (distances between ALL points
origDistances = np.empty((orig_rows, orig_rows))

items = range(len(data))
for i in items:
    for j in items:
        if i == j:
            origDistances[i][j] = .0
        else:
            origDistances[i][j] = euclidean(data[i], data[j])

#a = data[0]
#b = data[0]

if hasattr(data, "toarray"):
    data = data.toarray()

orig_dimension = len(data[0])

print '--------'
print type(data)
print np.shape(data)
print np.shape(data[0])
print np.shape(data[1])


def measureDistances(data, reduced, desc):


    a = np.shape(data)
    b = np.shape(reduced)
    if a[0] != b[0]:
        raise Exception("%s: same amount of instances required. data: %s, reduced: %s" % (desc, a,b))

    newDistancs = np.empty((b[0], b[0]))
    items = range(b[0])
    for i in items:
        for j in items:
            #if i == j:
            newDistancs[i][j] = 0
            #else:
            if i % 5 == 0 and j % 10 == 0:
                newDistancs[i][j] = euclidean(reduced[i], reduced[j])

    # compare item by item
    meanDistance = np.abs(np.mean(origDistances - newDistancs))

    #aa = np.shape(origDistances)
    #bb = np.shape(newDistancs)

    #d1 =  euclidean(data[0], data[1])
    #d2 =  euclidean(reduced[0], reduced[1])
    return meanDistance

def reduceAndMeasure(action, data):
    start = time.time()
    reduced = action(data)
    duration = time.time() - start

    return reduced, duration

def gaussianRP(data):
    rp = GaussianRandomProjection(n_components=new_dimension)
    return rp.fit_transform(data)

def sparseRP(data):
    rp = SparseRandomProjection(n_components=new_dimension)
    return rp.fit_transform(data)

def otherScikitImpls(data):
    rp = GaussianRandomProjection(n_components=new_dimension)
    m = rp._make_random_matrix(new_dimension, orig_dimension)
    m = np.mat(m)
    reduced = m * np.mat(data).transpose()

    reduced = reduced.transpose()
    return reduced

def custom1(data):
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


actions = {
    "gaussian RP": gaussianRP,
    "sparse RP": sparseRP,
    "other scikit": otherScikitImpls,
    "custom 1": custom1
}



results = dict()


x = np.arange(1,10)
for key in actions.iterkeys():

    durations = list()
    distances = list()
    for i in x:

        action = actions[key]

        reduced, d = reduceAndMeasure(action, data)
        dist = measureDistances(data, reduced, key)
        durations.append(d)
        distances.append(dist)

        #print "%s -> %s" % (key, duration)
        #print np.shape(reduced)

        #print
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
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08), fancybox=True, shadow=True, ncol=2)

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

plt.savefig( "%s/rp.png" % outputFolder, dpi=320, bbox_inches = "tight")


'''
reduced, dA = gaussianRP(data)
durationA.append(dA)

reduced2, dB = otherScikitImpls(data, dimensions)
durationB.append(dB)

reduced3, dC = sparseRP(data)
durationC.append(dC)

dist = euclidean(reduced[0], reduced[1])
dist2 = euclidean(reduced2[0], reduced2[1])
dist3 = euclidean(reduced3[0], reduced3[1])
distancesA.append(dist)
distancesB.append(dist2)
distancesC.append(dist3)

print "%s -> %s(%s) / %s(%s) / %s(%s)" % (origDist, np.abs(origDist-np.mean(distancesA)), np.sum(durationA) / i+1, np.abs(origDist-np.mean(distancesB)), np.sum(durationB) / i+1, np.abs(origDist-np.mean(distancesC)), np.sum(durationC) / i+1)
'''

'''
below = .0
null = .0
above = .0
for i in m:
    for item in i:
        if item < 0:
            below = below+1
        elif item > 0:
            above = above+1
        else:
            null = null + 1

sum = below + null + above

print below
print "%0.2f" % (below / sum)

print null
print "%0.2f" % (null / sum)

print above
print "%0.2f" % (above / sum)

print "----"
print sum

print m

'''
