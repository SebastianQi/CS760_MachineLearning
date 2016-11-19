from util import *
import numpy as np
import scipy.io.arff as sparff
import sys

def loadData(data_fname):
    # read the training data
    data, metadata = sparff.loadarff(data_fname)
    # local constants
    num_features = len(metadata.names())-1
    num_instances = len(data)
    y_info = metadata[metadata.names()[-1]]

    # convert data to a list of lists of integers
    X, Y = [], []
    for m in range(num_instances):
        # convert labels to 0-1 encoding
        Y.append(y_info[1].index(data[m][-1]))
        # create feature vector representation for each isntance
        featureVector = []
        for n in range(num_features):
            feature_info = metadata[metadata.names()[n]]
            featureVector.append(feature_info[1].index(data[m][n]))
        X.append(featureVector)

    # get the number of possible values of each feature
    numvals = []
    for n in range(len(metadata.names())):
        numvals.append(len(metadata[metadata.names()[n]][1]))
    return np.array(X), np.array(Y), metadata, numvals


def printGraph_NaiveBayes(metadata):
    num_features = len(metadata.names()) - 1
    for n in range(num_features):
        feature_info = metadata[metadata.names()[n]]
        if feature_info[0] == 'nominal':
            # the node and its parent (class for naive bayes)
            print('%s %s ' % (metadata.names()[n], metadata.names()[-1]))
        else:
            raise ValueError('Feature type must be "nominal"')
    print


def printSomeInstances(list, X):
    for rowIdx in list:
        print X[rowIdx,:]

def buildTreeAugBayesNet(X, Y, numVals):
    return 0


def buildNaiveBayesNet(X, Y, numVals):
    # compute the conditional probability
    P_Y = computeDistribution(Y, numVals[-1])
    P_XgY = computeP_XgY(X, Y, numVals)
    return P_Y, P_XgY

def printTestResults(Y_hat, Y_prob, Y_test, metadata):
    y_range = metadata[metadata.names()[-1]][1]
    for m in range(len(Y_test)):
        prediction = y_range[int(Y_hat[m])]
        truth = y_range[Y_test[m]]
        print('%s %s %.12f' % (prediction.strip('"\''), truth.strip('"\''), Y_prob[m]))
    hits = np.sum(np.around(Y_hat) == Y_test)
    print ('\n%d' % hits)

def naiveBayesPredict(x_new, P_Y, P_XgY, numVals_Y = 2):
    probs = np.zeros(numVals_Y,)
    for y_val in range(numVals_Y):
        probs[y_val] = P_Y[y_val]
        for n in range(len(x_new)):
            temp = P_XgY[y_val][n]
            probs[y_val] *= temp[x_new[n]]
    # get the idx and the value of the max
    prediction_distribution = np.divide(probs, np.sum(probs))
    predictedClass = np.argmax(prediction_distribution)
    predictedProbability = prediction_distribution[predictedClass]
    return predictedClass, predictedProbability

def computePredictions_NaiveBayes(X_test, P_Y, P_XgY):
    Y_hat, Y_prob = np.zeros(np.shape(X_test)[0],), np.zeros(np.shape(X_test)[0],)
    # loop over rows, make prediction
    for m in range(np.shape(X_test)[0]):
        Y_hat[m], Y_prob[m] = naiveBayesPredict(X_test[m,:], P_Y, P_XgY)
    return Y_hat, Y_prob


def computeP_XgY(X, Y, numVals):
    '''
    compute P(X = X_i | Y = Y_j), for all i and j
    '''
    P_XgY = [[],[]]
    # loop over all values of Y (binary)
    for val in range(numVals[-1]):
        X_sub, Y_sub = getDataSubset(X, Y, np.shape(X)[1], val)
        # loop over all features, to compute P(X = X_i | Y = val)
        for n in range(np.shape(X)[1]):
            P_XgY[val].append(computeDistribution(X_sub[:,n], numVals[n]))
    return P_XgY

def getDataSubset(X, Y, idx_feature, val_feature):
    '''
    Reduce X and Y, by selecting {rows: X_idx == val_feature}
    :param X: data matrix
    :param Y: label vector
    :param idx_feature: target feature index
    :param val_feature: target feature value
    :return: reduce X and Y
    '''
    # select rows, w.r.t to a particular feature with a particular value
    if idx_feature == np.shape(X)[1]:
        idx_target = Y == val_feature
    else:
        idx_target = X[:, idx_feature] == val_feature
    # subset X and Y
    X_reduced = X[idx_target,:]
    Y_reduced = Y[idx_target]
    return X_reduced, Y_reduced

def computeDistribution(vector, numVals):
    '''
    Given a vector of discrete value and number of possible values, compute the distribution
    :param vector: a natural number valued vector
    :param numVals: the number of possible values
    :return: a probability distribution
    '''
    counts = getFrequency(vector, numVals)
    # MAP estimation with laplace smoothing (PSEUDO_COUNTS == 1)
    counts += PSEUDO_COUNTS
    distribution = 1.0 * counts / np.sum(counts)
    return distribution

def getFrequency(vector, numVals):
    counts = np.zeros(numVals,)
    for value in vector:
        counts[value] +=1
    return counts
