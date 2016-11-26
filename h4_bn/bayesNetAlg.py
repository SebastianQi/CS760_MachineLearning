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




def buildTreeAugBayesNet(X, Y, numVals):
    return 0


def computeTreeWeights(X, Y, numVals):
    # compute the base rate of Y & P(X = X_i|Y) for all i
    P_Y = computeDistribution(Y, numVals[-1])
    P_XgY = computeP_XgY(X, Y, numVals)

    # compute P(X_i,X_j | Y) for all i,j
    P_XXgY = computeP_XX_Y(X, Y, numVals)
    P_XXY = computeP_XXY(X, Y, numVals)

    return P_Y, P_XgY, P_XXgY, P_XXY


def computeP_XX_Y(X, Y, numVals):
    '''
    :return: P_XXgY - Probability (X1 = xi, X2 = xj | Y = yk) for all i j k
        - P_XXgY is a list of list of list with structure : [y][X1][X2]
        - I did not compute diag terms, please use P_XgY directly
        - I believe only upper triangular entries are needed
    '''
    P_XXgY = []
    # loop over all possible y values
    for y_val in range(numVals[-1]):
        # condition on a particular y value
        X_red, Y_red = getDataSubset(X, Y, np.shape(X)[1], y_val)
        # compute P(Xi, Xj| Y = y)
        P_XXgY.append(computeP_XX(X_red, numVals))
    return P_XXgY


def computeP_XX(X, numVals):

    def computeP_XiXj(Xi, Xj, numVals_i, numVals_j):
        counts = np.zeros((numVals_i, numVals_j))
        # loop over all combinations of feature values (for Xi and Xj)
        for xi in range(numVals_i):
            for xj in range(numVals_j):
                # loop over all instances to accumulate the counts
                for m in range(len(Xi)):
                    if (Xi[m] == xi) and (Xj[m] == xj): counts[xi][xj] += 1
        P_XiXj = np.divide(counts, len(Xi))
        return P_XiXj

    N = np.shape(X)[1] -1
    P_XX = createListOfList(N, N)
    for i in range(N):
        for j in range(N):
            if i != j:
                P_XX[i][j] = computeP_XiXj(X[:,i], X[:,j], numVals[i],numVals[j])
    return P_XX


def computeP_XXY(X, Y, numVals):
    '''
    :return: a tensor of (triple) joint distribution
    '''
    joint_XXY = np.zeros((np.shape(X)[1],np.shape(X)[1], numVals[-1]))
    # loop over Y_k
    # for k in numVals[-1]:
    for k in range(numVals[-1]):
        # loop over all features
        for i in range(np.shape(X)[1]):
            # loop over all features
            for j in range(np.shape(X)[1]):
                if i != j:
                    print i, j
                    joint_XXY[i,j,k] = computeP_XiXjYk(X, i, j, numVals, Y, k)

        sys.exit('STOP')
    print joint_XXY
    return joint_XXY


def computeP_XiXjYk(X, i, j, numVals, Y, k):
    counts = np.zeros((numVals[i],numVals[j]))
    # xi, xj = numVals[i], numVals[j]
    for xi in range(numVals[i]):
        for xj in range(numVals[j]):
            # loop over instances
            for m in range(np.shape(X)[0]):
                if X[m,i] == xi and X[m,j] == xj and Y[m] == k:
                    counts[i,j]+=1
    distribution = np.divide(1.0 * counts, np.shape(X)[0])
    print distribution
    sys.exit('STOP')
    return distribution


def buildNaiveBayesNet(X, Y, numVals):
    # compute the base rate of Y & P(X = X_i|Y) for all i
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
    # MAP estimation with smoothing
    counts += PSEUDO_COUNTS
    distribution = 1.0 * counts / np.sum(counts)
    return distribution


def getFrequency(vector, numVals):
    counts = np.zeros(numVals,)
    for value in vector:
        counts[value] +=1
    return counts
