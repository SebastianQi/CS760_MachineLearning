import matplotlib.pyplot as plt
from kNN_alg import *
from util import *


# read input arguments
[fname_train, fname_test] = ['yeast_train.arff.txt', 'yeast_test.arff.txt']

# load data
X_train, Y_train, metadata = loadData(fname_train)
X_test, Y_test, _ = loadData(fname_test)
Y_range = metadata[metadata.names()[-1]][1]


Ks = [1, 30]
for i in range(len(Ks)):
    # get prediction
    _, Y_hats = testModel(X_train, Y_train, X_test, Y_test, Y_range, Ks[i], False)

    # compute the confusion matrix
    confusionMatrix = np.zeros((len(Y_range),len(Y_range)))
    for i in range(len(Y_test)):
        idx_pred = Y_range.index(Y_hats[i])
        idx_true = Y_range.index(Y_test[i])
        confusionMatrix[idx_pred, idx_true] += 1

    # show image
    plt.figure(i)
    plt.imshow(confusionMatrix, interpolation='nearest')
    plt.grid(True)
    plt.show()


