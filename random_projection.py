import numpy as np

import load_data as data
import iterations
import dimensions
import sys
from scipy.spatial.distance import euclidean

'''
    compare different implementations of random projection

    dataset:
    - https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/

    not sparse shape:   569,30
    sparse shape:       569,1369

    usage:

    compare by iterations
    $ python random_projection.py iterations 10

    compare by iterations with binary encoding
    $ python random_projection.py iterations 10 --encode

    compare by dimensions, 5 to 20 dimensions with stepsize 1
    $ python random_projection.py dimensions 5-20-1


    compare by dimensions with binary encoding, 50 to 500 dimensions with stepsize 50
    $ python random_projection.py dimensions 50-500-50 --encode


'''




len_args = len(sys.argv)
if len_args < 3 or len_args > 4:
    print "wrong parameter, see usage"
    exit()

params = sys.argv

mode = params[1]
config = params[2]
binary_encode = False

if "--encode" in params:
    binary_encode = True

print binary_encode

# amount of rows
if hasattr(data, "toarray"):
    data = data.toarray()

data = data.load(binary_encode)

orig_shape = np.shape(data)

print orig_shape

orig_rows = orig_shape[0]
orig_dimension = orig_shape[1]


origDistances = np.empty((orig_rows, orig_rows))
r = range(orig_rows)
for i in r:
    for j in r:
        if i == j:
            origDistances[i][j] = .0
        else:
            origDistances[i][j] = euclidean(data[i], data[j])


if mode == "iterations":
    param = int(config)
    iterations.compare(origDistances, data, param, binary_encode)
elif mode == "dimensions":
    config = config.split("-")
    if len(config) != 3:
        print "wrong parameter, see usage"
        exit()

    start = int(config[0])
    stop = int(config[1])
    stepsize = int(config[2])
    d = np.arange(start, stop+stepsize, stepsize)
    dimensions.compare(origDistances, data, d, binary_encode)
else:
    print "wrong parameter, see usage"
    exit()


