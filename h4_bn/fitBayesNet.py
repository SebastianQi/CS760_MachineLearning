import numpy as np
import sys

from bayesNetAlg import *
from util import *
from prim import *

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
    MI, P_Y, P_XgY, P_XXgY, P_XXY = computeTreeWeights(X_train, Y_train, numVals)
    MI = copyUpperTolowerTrig(MI)
    MST = findMaxSpanningTree_prim(MI)

    print 'TAN structure:'
    idx = [i for i in range(len(MST))]
    MST
    for i in range(len(MST)):
        print idx[i],MST[i]
    sys.exit('STOP')
    # print graph

    # make prediction
    Y_hat, Y_prob = computePredictions_TAN(X_test, MST, P_Y, P_XgY, P_XXgY, P_XXY)
    
    # print test results

else:
    raise ValueError('option must be either "n" or "t"')

