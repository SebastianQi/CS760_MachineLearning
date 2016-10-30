import numpy as np
import scipy.io.arff as sparff

def loadData(data_):
    # read the training data
    data, metadata = sparff.loadarff(data_)
    M = len(metadata.names()) - 1
    N = len(data)
    # convert data into a matrix form
    X = np.empty((N, M))
    Y = []
    for n in range(N):
        for m in range(M):
            X[n,m] = data[n][m]
        Y.append(data[n][M])
    return X, np.array(Y), metadata