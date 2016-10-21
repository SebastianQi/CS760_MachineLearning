from kNN_alg import *
from util import *

# read input arguments
fname_train, fname_test, Ks = processInputArgs_kNN_select()

# load data
X_train, Y_train, metadata = loadData(fname_train)
X_test, Y_test, _ = loadData(fname_test)
Y_range = metadata[metadata.names()[-1]][1]

# tune K with LOOV
N_train = X_train.shape[0]
performance = np.zeros((len(Ks),))
for i in range(len(Ks)):
    for n in range(N_train):
        performance[i,] += tuneModel_loov(X_train, Y_train, n, Y_range, Ks[i])

    if Y_range == None:
        print('Mean absolute error for k = %d : %.16f'
          % (Ks[i], 1.0 * performance[i,] / len(Y_train)))
    else:
        print('Number of incorrectly classified instances for k = %d : %d'
              % (Ks[i], len(Y_train) - performance[i,]))

# pick the best k
if Y_range == None:
    bestK = Ks[np.argmin(performance)]
else:
    bestK = Ks[np.argmax(performance)]

# fit final model with the best K
print('Best k value : %d' % bestK)
performance, _ = testModel(X_train, Y_train, X_test, Y_test, Y_range, bestK)

if Y_range == None:
    print('Mean absolute error : %.15f' % performance)
    print('Total number of instances : %d' % len(Y_test))
else:
    accuracy = 1.0 * performance / len(Y_test)
    print('Number of correctly classified instances : %d' % (performance))
    print('Total number of instances : %d' % len(Y_test))
    print('Accuracy : %.16f' % accuracy)