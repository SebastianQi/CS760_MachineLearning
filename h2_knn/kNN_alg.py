import numpy as np
from collections import OrderedDict
import scipy.io.arff as sparff
import sys

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


def getMajorityClass(Y_predicted, Y_range):
    # initialize the counts in a ordered dictionary
    counts = []
    for y_label in Y_range:
        counts.append((y_label, 0))
    counts = OrderedDict(counts)
    # accumulate the counts
    for y_pred in Y_predicted:
        counts[y_pred] += 1
    # return the majority vote
    majVoteClass = max(counts, key=counts.get)
    return majVoteClass

def computeDistances_l2(x_test, X_train):
    N_train = X_train.shape[0]
    distances = np.zeros(N_train,)
    # # compute d(x_test, X_train), where d is l2 distance function
    for n in range(N_train):
        distances[n] = np.linalg.norm(x_test - X_train[n,:])
    return distances

def findKSmallest(distances, K):
    distances = np.column_stack((range(len(distances)), distances))
    list = []
    copy = distances
    min_idx = []
    firstIter = True
    while len(list) < K:
        if firstIter:
            firstIter = False
        else:
            copy = np.delete(copy, min_idx, 0)

        min_idx = np.argmin(copy[:, 1])

        list.append(copy[min_idx, 0].astype(int))

    return list

def kNN_classify_l2(x_test, X_train, Y_train, Y_range, K):

    distances = computeDistances_l2(x_test, X_train)
    # distances = np.round(distances, 18)
    idx_k = findKSmallest(distances, K)
    # # find k smallest indices
    # idx_k = distances.argsort()[:K]

    tempdist = distances[idx_k]
    templabel = Y_train[idx_k]

    moreidxs = distances.argsort()[:4]
    moredists = distances[moreidxs]
    more_equal = moredists[2] == moredists[3]
    morelabels = Y_train[moreidxs]

    # for regression, return local average | for binary classification  return majority vote
    if Y_range == None:
        return np.mean(Y_train[idx_k])
    else:
        return getMajorityClass(Y_train[idx_k], Y_range)


def testModel(X_train, Y_train, X_test, Y_test, Y_range, K, printResults = True):
    '''
    fit kNN model with where k is defined by the input.
    :param X_train: the training data in a matrix form (instances x features)
    :param Y_train: label vector
    :param X_test: the test data in a matrix form (instances x features)
    :param Y_test: label vector
    :param Y_range: the possible values of labels, None if continuous valued
    :param K: kNN parameter
    :param printResults: whether to print the prediction-actual pair during evaluation
    :return: predicted Y & MAE (if continuous) OR correct counts (if discrete)
    '''
    count = 0
    mean_abs_error = 0
    Y_HAT = []
    # predict y, for all rows in X_Test
    for n in range(len(Y_test)):
        y_test_cur = Y_test[n]
        Y_hat = kNN_classify_l2(X_test[n,], X_train, Y_train, Y_range, K)
        Y_HAT.append(Y_hat)
        # for regression, aggregate mean absolute error
        if Y_range == None:
            mean_abs_error += abs(Y_hat - Y_test[n])
        # for binary classification, aggregate the count of correctly classified instance
        else:
            if Y_hat == Y_test[n]:
                count += 1

        if printResults:
            if Y_range == None:
                print('Predicted value : %.6f\tActual value : %.6f' % (Y_hat, Y_test[n]))
            else:
                print('Predicted class : %s\tActual class : %s' % (Y_hat, Y_test[n]))

    if Y_range == None:
        mean_abs_error = 1.0 * mean_abs_error / len(Y_test)
        return mean_abs_error, Y_HAT
    else:
        return count, Y_HAT


def tuneModel_loov(X, Y, idx_loocv, Y_range, K):
    # leave one out
    x_test = X[idx_loocv, :]
    y_test = Y[idx_loocv]
    X_train = np.delete(X, idx_loocv, 0)
    Y_train = np.delete(Y, idx_loocv)
    # compute the predition
    y_predict = kNN_classify_l2(x_test, X_train, Y_train, Y_range, K)
    # for regression, return MAE | for binary classification case, return boolean
    if Y_range == None:
        return abs(y_predict - y_test)
    else:
        return y_predict == y_test

