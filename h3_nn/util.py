import sys

TYPE_NOMINAL = 'nominal'
TYPE_NUMERIC = 'numeric'

WEIGHTS_INIT_LB = -0.01
WEIGHTS_INIT_UB =  0.01

SMALL_NUM = 1e-10

def processInputArgs_nnet():
    numArgs = 5
    # process input arguments
    if len(sys.argv) == numArgs+1:
        lrate = float(str(sys.argv[1]))
        nHidden = int(str(sys.argv[2]))
        nEpochs = int(str(sys.argv[3]))

        fname_train = str(sys.argv[4])
        fname_test = str(sys.argv[5])
    else:
        raise ValueError('ERROR: This program takes input arguments in the following way: '
                 '\n\tnnet.py l h e <train-set-file> <test-set-file>\n')

    print('%.3f, %d, %d, %s, %s' % (lrate, nHidden, nEpochs, fname_train, fname_test))
    return lrate, nHidden, nEpochs, fname_train, fname_test

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


def countBaseRate(Yvec):
    counts = 0
    for y in Yvec:
        if y == 1 :
            counts +=1

    return 1.0 * counts / len(Yvec)



