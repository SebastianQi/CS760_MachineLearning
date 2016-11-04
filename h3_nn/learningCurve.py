import matplotlib.pyplot as plt
import numpy as np
from nn_alg import *

fname_train = 'heart_train.arff.txt'
fname_test  = 'heart_test.arff.txt'

LRATE = .1
N_HIDDEN = 20
NUM_EPOCHS = [1, 10, 100, 500]
SAMPLE_SIZE = 10

# load data
X_train, Y_train = loadData(fname_train)
X_test, Y_test   = loadData(fname_test)

# preallocate
err_train = np.zeros(len(NUM_EPOCHS))
err_test = np.zeros(len(NUM_EPOCHS))

# train the model
for i in range(len(NUM_EPOCHS)):
    hits_train = np.zeros(SAMPLE_SIZE)
    hits_test  = np.zeros(SAMPLE_SIZE)
    for s in range(SAMPLE_SIZE):
        # train the model
        weights = trainModel(X_train, Y_train, N_HIDDEN, LRATE, NUM_EPOCHS[i], 0)
        # evalute on the training set
        hits_train[s] = testModel(X_train, Y_train, weights, 0)
        # evalute on the test set
        hits_test[s] = testModel(X_test, Y_test, weights, 0)

    print 1.0 * hits_train / len(Y_train)
    print 1.0 * hits_test / len(Y_test)
    # compute the mean error
    hits_train_mean = np.mean(hits_train[s])
    hits_test_mean = np.mean(hits_test[s])
    err_train[i] = 1.0 * (len(Y_train) - hits_train_mean) / len(Y_train)
    err_test[i] = 1.0 * (len(Y_test)- hits_test_mean) / len(Y_test)

# plot the training versus test set error
plt.figure(1)
LW = 2.0
plt.plot(NUM_EPOCHS, err_train, marker='o', linewidth=LW)
plt.plot(NUM_EPOCHS, err_test, marker='x', linewidth=LW)
plt.legend(['training set error', 'test set error'], loc='upper right')
plt.title('%s' % fname_test)
plt.ylabel('Error')
plt.xlabel('Training Epochs')

plt.show()

