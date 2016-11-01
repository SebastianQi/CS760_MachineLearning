from nn_alg import *

import numpy as np
import sys

lrate, nHidden, nEpochs, fname_train, fname_test = processInputArgs_nnet()

# load data
X_train, Y_train, _ = loadData(fname_train)
X_test, Y_test, _ = loadData(fname_test)
pattern_type = 'xor'
# X_train, Y_train = loadSimpleData(pattern_type)
# X_test, Y_test = loadSimpleData(pattern_type)
inputDim = len(X_train[0])

# initialize the weights to uniform random values
wts = initWeights(inputDim, nHidden)
for e in range(nEpochs):
    # update weights w.r.t one sweep of the training data
    # model without hidden units
    if nHidden == 0:
        wts, error, counts = deltaLearn(X_train, Y_train, wts, lrate)
    # general multilayerd model
    elif nHidden > 0:
        wts, error, counts = backprop(X_train, Y_train, wts, lrate)
    else:
        raise ValueError('Number of hidden units need to be postiive.\n')

    if np.mod(e,100) == 0:
        print ('Trainging Epoch = %6.d, error_L1 = %.6f, numCorrect = %d, numIncorrect = %d'
               % (e, error, counts, len(Y_train) - counts))

# testing
testModel(X_test, Y_test, wts)