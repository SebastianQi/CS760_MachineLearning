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



def printTestResults(Y_hat, Y_prob, Y_test, metadata):
    y_range = metadata[metadata.names()[-1]][1]
    for m in range(len(Y_test)):
        prediction = y_range[int(Y_hat[m])]
        truth = y_range[Y_test[m]]
        print('%s %s %.12f' % (prediction.strip('"\''), truth.strip('"\''), Y_prob[m]))
    hits = np.sum(np.around(Y_hat) == Y_test)
    print ('\n%d' % hits)


def printDataMatrix(X):
    for m in range(np.shape(X)[0]):
        print m, X[m][:]
    sys.exit('STOP')

def createList_2d(M,N):
    list_2d = [[np.nan for n in range(N)] for m in range(M)]
    return list_2d

def createList_3d(M,N,K):
    list_3d = [[[np.nan for k in xrange(K)] for n in xrange(N)] for m in xrange(M)]
    return list_3d


def printSomeInstances(list, X):
    for rowIdx in list:
        print X[rowIdx,:]


def printUpperTriangularPart(N):
    for i in range(N):

        sys.stdout.write('%d: ' % i)
        for k in range(i):
            sys.stdout.write('  ')

        # print the upper triangular part, w/o the diag entries
        for j in range(i+1, N, 1):
            sys.stdout.write('%d ' % j)
        sys.stdout.write('\n')


def printMatrix(matrix):
    M,N = np.shape(matrix)
    for m in range(M):
        for n in range(N):
            sys.stdout.write('%.18f ' % matrix[m,n])
        sys.stdout.write('\n')