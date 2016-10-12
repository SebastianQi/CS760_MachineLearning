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


# # #
# temp_counts = {}
# for y_label in Y_range: temp_counts[y_label] = 0
# for y in Y_test:
#     temp_counts[y] += 1
#
# print temp_counts
# for y_label in Y_range:
#     print("%d = %s" %( Y_range.index(y_label), y_label))
#
# sys.exit('STOP')
# # #

Ks = [1, 30]
for i in range(len(Ks)):
    # get prediction
    accuracy, Y_hats = testModel(X_train, Y_train, X_test, Y_test, Y_range, Ks[i], False)

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


