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

# # Debug
# print np.shape(X)
# print np.shape(Y)
# print X[2,:]
# print Y

N = len(data)
M = len(data[0,:]) - 1

train = np.zeros((N*0.6,M+1))
test = np.zeros((N*0.2,M+1))
validate = np.zeros((N*0.2,M+1))

i_train = 0
i_test = 0
i_validate = 0

randOrder = np.random.permutation(N)

counter = 0
for i in randOrder:
  j = counter % 10
  counter += 1
  if j < 6:
    train[i_train,:] = data[i,:]
    i_train += 1
  elif j < 8:
    test[i_test,:] = data[i,:]
    i_test += 1
  else:
    validate[i_validate,:] = data[i,:]
    i_validate += 1

print np.shape(train)
print np.shape(test)
print np.shape(validate)

np.savetxt(filename + "_train.csv", train, delimiter=",")
np.savetxt(filename + "_test.csv", test, delimiter=",")
np.savetxt(filename + "_validate.csv", validate, delimiter=",")



