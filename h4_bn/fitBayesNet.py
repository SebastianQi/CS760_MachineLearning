import numpy as np
import sys

from bayesNetAlg import *
from util import *


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
    # build the max spanning tree w.r.t to conditional mutual info
    MST = computeTanStructure(X_train, Y_train, numVals)
    # print tree structure
    printGraph_TAN(MST, metadata)
    # compute the the conditional probability
    CPT = buildTreeAugBayesNet(X_train, Y_train, numVals, MST)
    # make prediction
    Y_hat, Y_prob = computePredictions_TAN(X_test, CPT, MST, numVals)
    printTestResults(Y_hat, Y_prob, Y_test, metadata)

else:
    raise ValueError('option must be either "n" or "t"')

