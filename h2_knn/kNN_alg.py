import numpy as np
import sys

def getMajorityClass(Y_predicted, Y_range, nearest_label):
    # initialize the counts
    counts = {}
    for y_label in Y_range:
        counts[y_label] = 0
    # accumulate the counts
    for y_pred in Y_predicted:
        counts[y_pred] += 1

    # # return the majority vote
    majorityVotes = [k for k, val in counts.items() if val == max(counts.values())]
    if len(majorityVotes) > 1:
        return nearest_label
    else:
        return majorityVotes[0]



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
    print('Parameter: K = %d' % K)
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
            print('%d: Actual: %s Predicted: %s' % (n+1, Y_test[n], Y_hat))

    if Y_range == None:
        mean_abs_error = 1.0 * mean_abs_error / len(Y_test)
        print('MAE = %f' % mean_abs_error)
        return mean_abs_error
    else:
        print('Number of correctly classified: %d Total number of test instances: %d'
          % (count, len(Y_test)))
        accuracy = 1.0 * count / len(Y_test)
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