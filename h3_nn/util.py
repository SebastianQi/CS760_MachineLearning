import sys

def processInputArgs_nnet():
    numArgs = 4
    # process input arguments
    if len(sys.argv) == numArgs+1:
        nHidden = int(str(sys.argv[1]))
        nEpochs = int(str(sys.argv[2]))
        fname_train = str(sys.argv[3])
        fname_test = str(sys.argv[4])
    else:
        sys.exit('ERROR: This program takes input arguments in the following way: '
                 '\n\tnnet.py h e <train-set-file> <test-set-file>')
    return nHidden, nEpochs, fname_train, fname_test

# def processInputArgs_kNN_select():
#     numArgs = 5
#     # process input arguments
#     if len(sys.argv) == numArgs+1:
#         fname_train = str(sys.argv[1])
#         fname_test = str(sys.argv[2])
#         [K1, K2, K3] = [int(str(sys.argv[3])), int(str(sys.argv[4])), int(str(sys.argv[5]))]
#     else:
#         sys.exit('ERROR: This program takes input arguments in the following way: '
#                  '\n\tpython kNN-select.py <train-set-file> <test-set-file> k1 k2 k3')
#     return fname_train, fname_test, [K1, K2, K3]