import sys

def processInputArgs_kNN():
    numArgs = 3
    # process input arguments
    if len(sys.argv) == numArgs+1:
        fname_train = str(sys.argv[1])
        fname_test = str(sys.argv[2])
        K = int(str(sys.argv[3]))
    else:
        sys.exit('ERROR: This program takes input arguments in the following way: '
                 '\n\tpython kNN.py <train-set-file> <test-set-file> k')
    return fname_train, fname_test, K

def processInputArgs_kNN_select():
    numArgs = 5
    # process input arguments
    if len(sys.argv) == numArgs+1:
        fname_train = str(sys.argv[1])
        fname_test = str(sys.argv[2])
        [K1, K2, K3] = [int(str(sys.argv[3])), int(str(sys.argv[4])), int(str(sys.argv[5]))]
    else:
        sys.exit('ERROR: This program takes input arguments in the following way: '
                 '\n\tpython kNN-select.py <train-set-file> <test-set-file> k1 k2 k3')
    return fname_train, fname_test, [K1, K2, K3]