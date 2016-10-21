from util import *
from kNN_alg import *

# read input arguments
fname_train, fname_test, K = processInputArgs_kNN()
# load data
X_train, Y_train, metadata = loadData(fname_train)
X_test, Y_test, _ = loadData(fname_test)
Y_range = metadata[metadata.names()[-1]][1]

print('k value : %d' % K)
performance, _ = testModel(X_train, Y_train, X_test, Y_test, Y_range, K)

if Y_range == None:
    print('Mean absolute error : %.16f' % performance)
    print('Total number of instances : %d' % len(Y_test))
else:
    accuracy = 1.0 * performance / len(Y_test)
    print('Number of correctly classified instances : %d' % (performance))
    print('Total number of instances : %d' % len(Y_test))
    print('Accuracy : %.16f' % accuracy)