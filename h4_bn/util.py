import sys
import numpy as np

SMALL_NUM = 1e-10
PSEUDO_COUNTS = 1
NUM_INPUT_ARGS = 3 

def getInputArgs():
    # process input arguments
    if len(sys.argv) == NUM_INPUT_ARGS+1:
        fname_train = str(sys.argv[1])
        fname_test = str(sys.argv[2])
        option = str(sys.argv[3])
        if not (option == 'n' or option == 't'):
            raise ValueError('The 3rd argument is either "n" or "t" ')
    else:
        raise ValueError('This program takes input arguments in the following way: '
                 '\n\tbayes <train-set-file> <test-set-file> <n|t>\n')
    return fname_train, fname_test, option

def printDataMatrix(X):
    for m in range(np.shape(X)[0]):
        print m, X[m][:]
    sys.exit('STOP')


def createListOfList(m,n):
    listOfList = [[0 for x in range(m)] for y in range(n)]
    return listOfList


def printSomeInstances(list, X):
    for rowIdx in list:
        print X[rowIdx,:]

