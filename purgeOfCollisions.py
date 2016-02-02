__author__ = 'peteflorence'

import numpy as np
import pylab as pl
import scipy.optimize as opt
import scipy.io as sio


def getData(name):
    data = np.genfromtxt(name, delimiter=',')
    # Returns column matrices
    #X = data[:,0:-1]
    #Y = data[:,-1]
    return data

filename = 'sampleML_less_aggressive'
data = getData(filename + ".csv")
print np.shape(data)


def purgeOfCollisions(data):
    numDeleted = 0
    for i, rowvalue in enumerate(data):
        for j, value in enumerate(rowvalue[:-1]):
            if (data[i-numDeleted,j] == 0):
                data = np.delete(data, (i-numDeleted), axis=0)
                numDeleted += 1
    return data

data = purgeOfCollisions(data)
print np.shape(data)
np.savetxt(filename + "_purged.csv", data, delimiter=",")