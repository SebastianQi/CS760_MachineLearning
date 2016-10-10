import numpy as np
import sys

def getMajorityClass(y_partial, Y_range):
    [counts1, counts2] = [0,0]
    for y in y_partial:
        if y == Y_range[0]:
            counts1 += 1
        elif y == Y_range[1]:
            counts2 += 1
        else:
            raise
    if counts1 >= counts2:
        return Y_range[0]
    else:
        return Y_range[1]
    # TODO : handle tie


def kNN_classify_l2(x_test, X_train, Y_train, Y_range, K):
    N_train = X_train.shape[0]
    distances = np.zeros(N_train,)
    # compute d(x_test, X_train), where d is l2 distance function
    for n in range(N_train):
        distances[n] = np.linalg.norm(x_test - X_train[n,:])
    # find k smallest indices
    idx = np.argpartition(distances, K)[:K]

    # for regression, average k nearst elements
    if Y_range == None:
        prediction = np.mean(Y_train[idx])
    # for binary classification  get majority vote for K nearst neighbours
    else:
        prediction = getMajorityClass(Y_train[idx], Y_range)
    return prediction


def testModel(X_train, Y_train, X_test, Y_test, Y_range, K, printResults = True):
    print('Parameter: K = %d' % K)
    count = 0
    mae = 0
    # predict y, for all rows in X_Test
    for n in range(X_test.shape[0]):
        Y_predict = kNN_classify_l2(X_test[n,], X_train, Y_train, Y_range, K)
        # for regression, aggregate mean absolute error
        if Y_range == None:
            mae += sum(abs(Y_predict - Y_test[n]))
        # for binary classification, aggregate the count of correctly classified instance
        else:
            if Y_predict == Y_test[n]:
                count += 1

        if printResults:
            print('%d: Actual: %s Predicted: %s' %(n+1, Y_test[n], Y_predict))


    if Y_range == None:
        print('Cumulative MAE = ' % mae)
        return mae
    else:
        print('Number of correctly classified: %d Total number of test instances: %d'
          % (count,len(Y_test)))
        error = 1 - 1.0*count/len(Y_test)
        return error


def tuneModel_loov(X_train, Y_train, x_test, y_test, Y_range, K):
    y_predict = kNN_classify_l2(x_test, X_train, Y_train, Y_range, K)
    # for regression, return true or false
    if Y_range == None:
        return sum(abs(y_predict - y_test))
    # for binary classification case, return mean absolute error
    else:
        if y_predict == y_test:
            return 1
        else:
            return 0