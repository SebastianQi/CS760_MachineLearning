from util import *
from nn_alg import *

import numpy as np
import sys

nHidden, nEpochs, fname_train, fname_test = processInputArgs_nnet()

# load data
X_train, Y_train, metadata = loadData(fname_train)
X_test, Y_test, _ = loadData(fname_test)

initNeuralNetwork(metadata, nHidden)

trainModel(fname_train)


