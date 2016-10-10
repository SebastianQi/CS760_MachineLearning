import numpy as np
import scipy.io.arff as sparff
import sys

from util import *
##################################################################################################
############################### HELPER FUNCTIONS #################################################
##################################################################################################
def processInputArgs():
    # process input arguments
    if len(sys.argv) == 4:
        fname_train = str(sys.argv[1])
        fname_test = str(sys.argv[2])
        K = int(str(sys.argv[3]))
    else:
        sys.exit('ERROR: This program takes input arguments in the following way: '
                 '\n\tpython kNN.py <train-set-file> <test-set-file> k')
    return fname_train, fname_test, K


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




def getKLargestElements(array, K):
    ind_largestK = np.argpartition(array, -K)[-K:]
    return ind_largestK


def getMajorityClass(y_partial):
    [counts1, counts2] = [0,0]

    for y in y_partial:
        if y == y_range[0]:
            counts1 += 1
        else:
            counts2 += 1

    if counts1 >= counts2:
        return y_range[0]
    else:
        return y_range[1]
    # TODO : handle tie case



def kNN_classify_l2(x_test, X_train, Y_train, K):
    (N_train, M) = X_train.shape
    distances = np.zeros(N_train,)

    for n in range(N_train):
        distances[n] = np.linalg.norm(x_test - X_train[n,:])

    idx = getKLargestElements(distances,K)
    prediction = getMajorityClass(Y_train[idx])
    return prediction


def printPerformance():
    # predict the label of y
    N_test = X_test.shape[0]
    count = 0
    for n in range(N_test):
        Y_predict = kNN_classify_l2(X_test[n,], X_train, Y_train, K)

        if Y_predict == Y_test[n]:
            count += 1

        print('%d: Actual: %s Predicted: %s' %(n+1, Y_test[n], Y_predict))

    print('Number of correctly classified: %d Total number of test instances: %d'
          % (count,len(Y_test)))


##################################################################################################
############################### "Main" program #################################################
##################################################################################################
# read input arguments
fname_train, fname_test, K = processInputArgs()
# load data
X_train, Y_train, metadata = loadData(fname_train)
X_test, Y_test, _ = loadData(fname_test)
y_range = metadata[metadata.names()[-1]][1]


printPerformance()



