__author__ = 'peteflorence'

import numpy as np
import pylab as pl
import scipy.optimize as opt
import scipy.io as sio


def getData(name):
    data = np.genfromtxt('sampleML.csv', delimiter=',')
    # Returns column matrices
    X = data[:,0:-1]
    Y = data[:,-1]
    return X, Y

filename = 'sampleML'
X, Y = getData(filename + ".csv")

# # Debug
# print np.shape(X)
# print np.shape(Y)
# print X[2,:]
# print Y

N = len(Y)
M = len(X[0,:])

trainX = np.zeros((N*0.6,M))
trainY = np.zeros((N*0.6,1))

testX = np.zeros((N*0.2,M))
testY = np.zeros((N*0.2,1))

validateX = np.zeros((N*0.2,M))
validateY = np.zeros((N*0.2,1))


i_train = 0
i_test = 0
i_validate = 0

randOrder = np.random.permutation(N)

counter = 0
for i in randOrder:
  j = counter % 10
  counter += 1
  if j < 6:
    trainX[i_train,:] = X[i,:]
    trainY[i_train] = Y[i]
    i_train += 1
  elif j < 8:
    testX[i_test,:] = X[i,:]
    testY[i_test] = Y[i]
    i_test += 1
  else:
    validateX[i_validate,:] = X[i,:]
    validateY[i_validate] = Y[i]
    i_validate += 1

print np.shape(trainX)
print np.shape(trainY)
print np.shape(testX)
print np.shape(testY)
print np.shape(validateX)
print np.shape(validateY)

np.savetxt(filename + "_trainX.csv", trainX, delimiter=",")
np.savetxt(filename + "_trainY.csv", trainY, delimiter=",")
np.savetxt(filename + "_testX.csv", testX, delimiter=",")
np.savetxt(filename + "_testY.csv", testY, delimiter=",")
np.savetxt(filename + "_validateX.csv", validateX, delimiter=",")
np.savetxt(filename + "_validateY.csv", validateY, delimiter=",")



