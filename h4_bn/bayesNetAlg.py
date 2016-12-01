from util import *
from prim import *

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



## TAN


def computePredictions_TAN(X_test, CPT, parents, numVals):
    [M,N] = np.shape(X_test)
    Y_hat, Y_prob = np.zeros(M,),np.zeros(M,)
    for m in range(M):
        p_y = np.zeros(numVals[-1],)
        for y in range(numVals[-1]):
            p_y[y] = computeP_y_given_allx(y, X_test[m,:], CPT, parents)

        p_y = np.divide(p_y, np.sum(p_y))
        Y_prob[m] = max(p_y)
        Y_hat[m] = np.argmax(p_y)

    return Y_hat, Y_prob

def computeP_y_given_allx(y, x, CPT, parents):
    p = CPT[-1][y]

    for n in range(len(x)):

        xi = x[n]
        if parents[n] == None:
            P_Xi_g_Xj_Y = CPT[n][y][xi]
        else:
            xj = x[parents[n]]
            P_Xi_g_Xj_Y = CPT[n][xi][xj][y]

        p *= P_Xi_g_Xj_Y
    return p


def buildTreeAugBayesNet(X, Y, numVals, parents):
    CPT = []
    for n in range(np.shape(X)[1]):
        CPT.append(computeCPT_Xi(n, X, Y, numVals, parents[n]))
    CPT.append(computeDistribution(Y, numVals[-1]))
    return CPT


def computeCPT_Xi(f_idx, X, Y, numVals, parent):

    def getP_XigY(f_idx, P_XgY, numVals):
        P_XigY = []
        for k in range(numVals[-1]):
            P_XigY.append(P_XgY[k][f_idx])
        return P_XigY  # [y][x]

    P_XgY = computeP_XgY(X, Y, numVals)
    # for the root of the tree
    if parent == None:
        return getP_XigY(f_idx, P_XgY, numVals)

    # for a generic node in the tree
    else:
        P_XiXjY = []
        for k in range(numVals[-1]):
            P_XiXjYk = computeP_XiXjYk(X[:, f_idx], X[:, parent],
                                       numVals[f_idx], numVals[parent], Y, k)
            P_XiXjY.append(P_XiXjYk)

        P_XigY = getP_XigY(f_idx, P_XgY, numVals)

        CPT_Xi = computeP_XigXjY(f_idx, parent, numVals, P_XiXjY, P_XigY)

        return CPT_Xi



def computeP_XigXjY(i, j, numVals, P_XiXjY, P_XigY):
    P_Xi_g_XjY = createList_3d(numVals[i], numVals[j], numVals[-1])
    for k in range(numVals[-1]):
        for vi in range(numVals[i]):
            for vj in range(numVals[j]):
                P_Xi_g_XjY[vi][vj][k] = P_XiXjY[k][vi][vj] / P_XigY[k][vi]

    return P_Xi_g_XjY


def printGraph_TAN(parents, metadata):
    featureNames = metadata.names()
    for n in range(len(featureNames) -1):
        # print the immediate parent, follow by Y
        if parents[n] == None:
            print ('%s %s' % (featureNames[n], featureNames[-1]))
        else:
            print ('%s %s %s'% (featureNames[n], featureNames[parents[n]], featureNames[-1]))
    print

def computeTanStructure(X_train, Y_train, numVals):
    MI, P_Y, P_XgY, P_XXgY, P_XXY = computeTreeWeights(X_train, Y_train, numVals)
    MI = copyUpperTolowerTrig(MI)
    MST = findMaxSpanningTree_prim(MI)
    return MST


def computeTreeWeights(X, Y, numVals):
    # compute the base rate of Y & P(X = X_i|Y) for all i
    P_Y = computeDistribution(Y, numVals[-1])
    P_XgY = computeP_XgY(X, Y, numVals)
    # compute P(X_i,X_j | Y) for all i,j
    P_XXgY = computeP_XX_Y(X, Y, numVals)
    P_XXY = computeP_XXY(X, Y, numVals)
    # compute the weights
    MI = computeMI(P_XgY, P_XXgY, P_XXY, numVals)

    return MI, P_Y, P_XgY, P_XXgY, P_XXY


def computeMI(P_XgY, P_XXgY, P_XXY, numVals):

    numVal_y = numVals[-1]
    N = len(numVals)-1
    mutualInfo = np.zeros((N,N))
    # compute the upper triangular part of the mutual information matrix
    for i in range(N):
        for j in range(i+1,N,1):
        # for j in range(N):

            for k in range(numVal_y):
                # compute mutual information
                mutualInfo[i, j] += computeMI_XiXjgYk(P_XXY[k][i][j], P_XXgY[k][i][j], P_XgY[k][i],
                                                      P_XgY[k][j], numVals[i], numVals[j])
    # assign the diag terms to be zeros
    for n in range(N):
        mutualInfo[n,n] = -1
    return mutualInfo

def computeMI_XiXjgYk(P_XiXjYk, P_XiXj_Yk, P_Xi_Yk, P_Xj_Yk, numVals_i, numVals_j):
    mi = 0
    for xi in range(numVals_i):
        for xj in range(numVals_j):
            mi += P_XiXjYk[xi][xj] * np.log2(P_XiXj_Yk[xi, xj] / (P_Xi_Yk[xi] * P_Xj_Yk[xj]))

    return mi


def computeP_XX_Y(X, Y, numVals):
    '''
    :return: P_XXgY - Probability (X1 = xi, X2 = xj | Y = yk) for all i j k
        - P_XXgY is a list of list of list with structure : [y][X1][X2]
        - I did not compute diag terms, please use P_XgY directly
        - I believe only upper triangular entries are needed
    '''

    def computeP_XX(X, numVals):
        # compute P(Xi, Xj) for a pair of i and j
        def computeP_XiXj(Xi, Xj, numVals_i, numVals_j):
            counts = np.zeros((numVals_i, numVals_j))
            # loop over all combinations of feature values (for Xi and Xj)
            for xi in range(numVals_i):
                for xj in range(numVals_j):
                    # loop over all instances to accumulate the counts
                    for m in range(len(Xi)):
                        if (Xi[m] == xi) and (Xj[m] == xj): counts[xi][xj] += 1
            # MAP estimation with smoothing
            counts += PSEUDO_COUNTS
            # TODO check denominator
            P_XiXj = np.divide(counts, len(Xi) + numVals_i * numVals_j)
            return P_XiXj

        # compute P(Xi, Xj) for all i and j
        N = np.shape(X)[1]

        P_XX = createList_2d(N, N)
        for i in range(N):
            for j in range(N):
                if i != j:
                    P_XX[i][j] = computeP_XiXj(X[:, i], X[:, j], numVals[i], numVals[j])
        return P_XX

    P_XXgY = []
    # loop over all possible y values
    for y_val in range(numVals[-1]):
        # condition on a particular y value
        X_red, Y_red = getDataSubset(X, Y, np.shape(X)[1], y_val)
        # compute P(Xi, Xj| Y = y)
        P_XXgY.append(computeP_XX(X_red, numVals))
    return P_XXgY



def computeP_XXY(X, Y, numVals):
    '''
    :return: a matrix of (triple) joint distribution (X1 x X2)
        - each entry is [Xi = xi][Xj = xj][Y = y]
    '''
    N = np.shape(X)[1]
    P_XXY = createList_3d(numVals[-1], N, N)
    # loop over all features: the upper triangular part
    for y_val in range(numVals[-1]):
        for i in range(N):
            for j in range(i+1, N, 1):
                if i != j:
                    P_XXY[y_val][i][j] = computeP_XiXjYk(X[:,i], X[:,j], numVals[i], numVals[j],
                                                         Y, y_val)
    return P_XXY


def computeP_XiXjYk(Xi, Xj, numVals_i, numVals_j, Y, y_val):
    M = len(Xi)
    counts = np.zeros((numVals_i, numVals_j))
    # loop over all possible values of X1, X2
    for xi in range(numVals_i):
        for xj in range(numVals_j):
            # loop over instances to get the counts
            for m in range(M):
                if Xi[m] == xi and Xj[m] == xj and Y[m] == y_val:
                    counts[xi, xj] += 1
    # smoothing
    counts += PSEUDO_COUNTS
    # TODO check denominator
    P_XiXjYk = np.divide(1.0 * counts, M + 2 * numVals_i * numVals_j)
    return P_XiXjYk



### Naive bayes

def printGraph_NaiveBayes(metadata):
    num_features = len(metadata.names()) - 1
    for n in range(num_features):
        feature_info = metadata[metadata.names()[n]]
        if feature_info[0] == 'nominal':
            # the node and its parent (class for naive bayes)
            print('%s %s' % (metadata.names()[n], metadata.names()[-1]))
        else:
            raise ValueError('Feature type must be "nominal"')
    print



def buildNaiveBayesNet(X, Y, numVals):
    # compute the base rate of Y & P(X = X_i|Y) for all i
    P_Y = computeDistribution(Y, numVals[-1])
    P_XgY = computeP_XgY(X, Y, numVals)
    return P_Y, P_XgY


def computePredictions_NaiveBayes(X_test, P_Y, P_XgY):
    def naiveBayesPredict(x_new, P_Y, P_XgY, numVals_Y=2):
        probs = np.zeros(numVals_Y, )
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


def computeDistribution(Xi, numVals):
    '''
    Given a vector of discrete value and number of possible values, compute the distribution
    :param Xi: a natural number valued vector
    :param numVals: the number of possible values
    :return: a probability distribution
    '''
    def getFrequency(vector, numVals):
        counts = np.zeros(numVals, )
        for value in vector: counts[value] += 1
        return counts

    counts = getFrequency(Xi, numVals)
    # MAP estimation with smoothing
    counts += PSEUDO_COUNTS
    distribution = 1.0 * counts / np.sum(counts)
    return distribution



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



