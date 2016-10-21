import matplotlib.pyplot as plt
from kNN_alg import *
from util import *
import sys

# read input arguments
[fname_train, fname_test] = ['yeast_train.arff.txt', 'yeast_test.arff.txt']

# load data
X_train, Y_train, metadata = loadData(fname_train)
X_test, Y_test, _ = loadData(fname_test)
Y_range = metadata[metadata.names()[-1]][1]


Ks = [1, 30]
for i in range(len(Ks)):
    # get prediction
    hits, Y_hats = testModel(X_train, Y_train, X_test, Y_test, Y_range, Ks[i], False)
    accuracy = 1.0 * hits / len(Y_test)
    # compute the confusion matrix
    confusionMatrix = np.zeros((len(Y_range),len(Y_range)))
    for n in range(len(Y_test)):
        idx_pred = Y_range.index(Y_hats[n])
        idx_true = Y_range.index(Y_test[n])
        confusionMatrix[idx_pred, idx_true] += 1

    # show image
    plt.figure(i)
    plt.title('Confusion Matrix - %s\n K = %d, accuracy = %f' % (fname_test, Ks[i], accuracy))
    plt.xlabel('Prediction')
    plt.ylabel('Truth')

    plt.xticks(range(len(Y_range)), list(Y_range))
    plt.yticks(range(len(Y_range)), list(Y_range))

    plt.imshow(confusionMatrix, interpolation='nearest')
    plt.colorbar()
    plt.clim(0, 90)
    plt.show()