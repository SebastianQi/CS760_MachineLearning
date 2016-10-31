from util import *
from nn_alg import *

import numpy as np
import sys

lrate, nHidden, nEpochs, fname_train, fname_test = processInputArgs_nnet()

# load data
X_train, Y_train, _ = loadData(fname_train)
X_test, Y_test, _ = loadData(fname_test)
# X_train, Y_train = loadSimpleData('xor')
inputDim = len(X_train[0])

# initialize the weights to uniform random values
wts = initWeights(inputDim, nHidden)

# model without hidden units
if nHidden == 0:
    for e in range(nEpochs):
        if np.mod(e,100) == 0:
            print ('Trainging Epoch = %d' % e)
        # update weights w.r.t one sweep of the training data
        wts = deltaLearn(X_train, Y_train, wts, lrate)

# general multilayerd model
elif nHidden > 0:
    for e in range(nEpochs):
        if np.mod(e,100) == 0:
            print ('Trainging Epoch = %d' % e)
        # update weights w.r.t one sweep of the training data
        wts = backprop(X_train, Y_train, wts, lrate)
else:
    raise ValueError('Number of hidden units need to be postiive.\n')



# testing

testModel(X_train, Y_train, wts)


