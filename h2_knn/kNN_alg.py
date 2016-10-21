import numpy as np
from collections import OrderedDict
import sys

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

def kNN_classify_l2(x_test, X_train, Y_train, Y_range, K):
    N_train = X_train.shape[0]
    distances = np.zeros(N_train,)
    # compute d(x_test, X_train), where d is l2 distance function
    for n in range(N_train):
        distances[n] = np.linalg.norm(x_test - X_train[n,:])
    # find k smallest indices
    idx_k = distances.argsort()[:K]

    if Y_range == None:
        # for regression, average k nearst elements
        prediction = np.mean(Y_train[idx_k])
    else:
        # for binary classification  get majority vote for K nearst neighbours
        prediction = getMajorityClass(Y_train[idx_k], Y_range)
    return prediction


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
        Y_hat = kNN_classify_l2(X_test[n,], X_train, Y_train, Y_range, K)
        Y_HAT.append(Y_hat)
        # for regression, aggregate mean absolute error
        if Y_range == None:
            mean_abs_error += abs(Y_hat - Y_test[n])
        # for binary classification, aggregate the count of correctly classified instance
        else:
            if Y_hat == Y_test[n]: count += 1

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

def tuneModel_loov(X, Y, idx_del, Y_range, K):
    # compute the predition
    x_test = X[idx_del, :]
    y_test = Y[idx_del]
    X_train = np.delete(X, idx_del, 0)
    Y_train = np.delete(Y, idx_del)
    y_predict = kNN_classify_l2(x_test, X_train, Y_train, Y_range, K)

    if Y_range == None:
        # for regression, return MAE
        abs_error = abs(y_predict - y_test)
        return abs_error
    else:
        # for binary classification case, return boolean
        return y_predict == y_test
