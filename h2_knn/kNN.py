import numpy as np
import sys
from util import *
##################################################################################################
############################### HELPER FUNCTIONS #################################################
##################################################################################################

def getMajorityClass(y_partial):
    [counts1, counts2] = [0,0]
    for y in y_partial:
        if y == y_range[0]:
            counts1 += 1
        elif y == y_range[1]:
            counts2 += 1
        else:
            raise
    if counts1 >= counts2:
        return y_range[0]
    else:
        return y_range[1]
    # TODO : handle tie


def kNN_classify_l2(x_test, X_train, Y_train, K):
    N_train = X_train.shape[0]
    distances = np.zeros(N_train,)
    # compute d(x_test, X_train), where d is l2 distance function
    for n in range(N_train):
        distances[n] = np.linalg.norm(x_test - X_train[n,:])
    # find k smallest indices
    idx = np.argpartition(distances, K)[:K]
    # get majority vote for K nearst neighbours
    prediction = getMajorityClass(Y_train[idx])
    return prediction


def printPerformance():
    count = 0
    # predict y, for all rows in X_Test
    for n in range(X_test.shape[0]):
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



