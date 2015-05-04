import numpy as np
import implementation as impl
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

def compare(origDistances, data, iterations, encode):
    origShape = np.shape(data)
    print iterations

    results = dict()
    new_dimension = 10
    orig_dimension = origShape[1]

    x = np.arange(1,iterations, 1)
    for key in impl.actions.iterkeys():
        print key
        durations = list()
        distances = list()
        di = list()
        du = list()
        for i in x:
            print " %s" % i
            action = impl.actions[key]

            reduced, d = impl.reduceAndMeasure(action, data, orig_dimension, new_dimension)
            dist = impl.measureDistances(origDistances, data, reduced, key)
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

    e = ""
    if encode:
        e = "encode"

    plt.savefig( "%s/rp_iterations_%s_%s.png" % (outputFolder, iterations, e), dpi=320, bbox_inches = "tight")

