import numpy as np
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
hits_tune = np.zeros((len(Ks),N_train))
for i in range(len(Ks)):
    for n in range(N_train):
        hits_tune[i,n] = tuneModel_loov(np.delete(X_train, n, 0), np.delete(Y_train, n),
                                       X_train[n,:], Y_train[n], Y_range, Ks[i])
# pick K that maximize tunning accuracy
accuracy_tune = np.mean(hits_tune,1)
bestK = Ks[np.argmax(accuracy_tune)]

# fit final model with the best K
accuracy_test = testModel(X_train, Y_train, X_test, Y_test, Y_range, bestK)


# ##
# import matplotlib.pyplot as plt
# plt.figure(1)
# plt.plot(Ks, accuracy)
#
# plt.title('%s' % fname_test)
# plt.ylabel('Test set classification accuracy')
# plt.xlabel('K')
# plt.show()

