import numpy as np
from bayesNetAlg import *
from util import *
import sys

# load data
fname_train, fname_test, option = getInputArgs()
X_train, Y_train, metadata, numVals = loadData(fname_train)
X_test, Y_test, _, _ = loadData(fname_test)


if option == 'n':
    # print the structure of the naive bayes
    printGraph_NaiveBayes(metadata)
    # train naive bayes classifer
    P_Y, P_XgY = buildNaiveBayesNet(X_train, Y_train, numVals)
    Y_hat, Y_prob = computePredictions_NaiveBayes(X_test, P_Y, P_XgY)
    printTestResults(Y_hat, Y_prob, Y_test, metadata)

elif option == 't':
    # train a tree agumented bayes classifer
    buildTreeAugBayesNet(X_train, Y_train, numVals)

else:
    raise ValueError('option must be either "n" or "t"')

