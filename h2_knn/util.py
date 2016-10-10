import numpy as np
import scipy.io.arff as sparff
import sys

TYPE_NOMINAL = 'nominal'
TYPE_NUMERIC = 'numeric'
NAME_RESPONSE = 'response'


def processInputArgs_kNN():
    numArgs = 3
    # process input arguments
    if len(sys.argv) == numArgs+1:
        fname_train = str(sys.argv[1])
        fname_test = str(sys.argv[2])
        K = int(str(sys.argv[3]))
    else:
        sys.exit('ERROR: This program takes input arguments in the following way: '
                 '\n\tpython kNN.py <train-set-file> <test-set-file> k')
    return fname_train, fname_test, K

def processInputArgs_kNN_select():
    numArgs = 5
    # process input arguments
    if len(sys.argv) == numArgs+1:
        fname_train = str(sys.argv[1])
        fname_test = str(sys.argv[2])
        [K1, K2, K3] = [int(str(sys.argv[3])), int(str(sys.argv[4])), int(str(sys.argv[5]))]
    else:
        sys.exit('ERROR: This program takes input arguments in the following way: '
                 '\n\tkNN-select <train-set-file> <test-set-file> k1 k2 k3')
    return fname_train, fname_test, [K1, K2, K3]

def loadData(data_):
    # read the training data
    data, metadata = sparff.loadarff(data_)
    M = len(metadata.names()) - 1
    N = len(data)
    # convert data into a matrix form
    X = np.empty((N, M))
    X[:] = np.NAN
    Y = []
    for n in range(N):
        for m in range(M):
            X[n,m] = data[n][m]
        Y.append(data[n][M])
    return X, np.array(Y), metadata



def showDataInfo(metadata):
    M = len(metadata.names()) - 1
    for m in range(M):
        print("X_%d = <%s>" % (m, metadata.names()[m]))
    print ("Y = <%s> is <%s>" % (metadata.names()[-1], metadata.types()[-1]))


