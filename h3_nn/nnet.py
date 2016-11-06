from nn_alg import *

import numpy as np
import sys

# load data
lrate, nHidden, nEpochs, fname_train, fname_test = processInputArgs_nnet()
X_train, Y_train = loadData(fname_train)
X_test, Y_test   = loadData(fname_test)

# pattern_type = 'xor'
# X_train, Y_train = loadSimpleData(pattern_type)
# X_test, Y_test = loadSimpleData(pattern_type)

# train the model
weights = trainModel(X_train, Y_train, nHidden, lrate, nEpochs)
# evalute on the test set
TP, TN, FP, FN, _ = testModel(X_test, Y_test, weights)