import sys

THRESHOLD = .5
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




