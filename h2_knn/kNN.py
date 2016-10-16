import sys
from util import *
from kNN_alg import *

# read input arguments
fname_train, fname_test, K = processInputArgs_kNN()
# load data
X_train, Y_train, metadata = loadData(fname_train)
X_test, Y_test, _ = loadData(fname_test)
Y_range = metadata[metadata.names()[-1]][1]

testModel(X_train, Y_train, X_test, Y_test, Y_range, K)