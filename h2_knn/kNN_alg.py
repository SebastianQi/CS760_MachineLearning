import numpy as np
from collections import OrderedDict
import sys

def getMajorityClass(Y_predicted, Y_range, nearest_label):
    # initialize the counts in a ordered dictionary
    counts = []
    for y_label in Y_range:
        counts.append((y_label,0))
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
    idx = np.argpartition(distances, K)[:K]
    nearest_idx = np.argpartition(distances, 1)[:1]

    if Y_range == None:
        # for regression, average k nearst elements
        prediction = np.mean(Y_train[idx])
    else:
        # for binary classification  get majority vote for K nearst neighbours
        prediction = getMajorityClass(Y_train[idx], Y_range, Y_train[nearest_idx][0])
    return prediction


def testModel(X_train, Y_train, X_test, Y_test, Y_range, K, printResults = True):
    print('k value : %d' % K)
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
            if Y_hat == Y_test[n]:
                count += 1

        if printResults:
            if Y_range == None:
                print('Predicted value : %.6f\tActual value : %.6f' % (Y_hat, Y_test[n]))
            else:
                print('Predicted value : %s\tActual value : %s' % (Y_hat, Y_test[n]))

    if Y_range == None:
        mean_abs_error = 1.0 * mean_abs_error / len(Y_test)
        print('Mean absolute error : %.16f' % mean_abs_error)
        print('Total number of instances : %d' % len(Y_test))
        return mean_abs_error, Y_HAT
    else:
        accuracy = 1.0 * count / len(Y_test)
        print('Number of correctly classified instances : %d ' % (count))
        print('Total number of instances : %d' % len(Y_test))
        print('Accuracy : %.16f' % accuracy)
        return accuracy, Y_HAT


def tuneModel_loov(X_train, Y_train, x_test, y_test, Y_range, K):
    y_predict = kNN_classify_l2(x_test, X_train, Y_train, Y_range, K)

    if Y_range == None:
        # for regression, return true or false
        abs_error = abs(y_predict - y_test)
        return abs_error
    else:
        # for binary classification case, return mean absolute error
        if y_predict == y_test:
            return 1
        else:
            return 0