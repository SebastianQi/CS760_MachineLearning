import matplotlib.pyplot as plt
from nn_alg import *

fname_train = 'heart_train.arff.txt'
fname_test  = 'heart_test.arff.txt'
# fname_train = 'lymph_train.arff.txt'
# fname_test  = 'lymph_test.arff.txt'

LRATE = .1
N_HIDDEN = 0
NUM_EPOCHS = [1, 10, 100, 500]
SAMPLE_SIZE = 1

# load data
X_train, Y_train = loadData(fname_train)
X_test, Y_test   = loadData(fname_test)

# preallocate
err_train = np.zeros(len(NUM_EPOCHS))
err_test = np.zeros(len(NUM_EPOCHS))

# loop over max training epochs
for i in range(len(NUM_EPOCHS)):
    TP_train = np.zeros(SAMPLE_SIZE); FP_train = np.zeros(SAMPLE_SIZE)
    TN_train = np.zeros(SAMPLE_SIZE); FN_train = np.zeros(SAMPLE_SIZE)
    TP_test = np.zeros(SAMPLE_SIZE);  FP_test = np.zeros(SAMPLE_SIZE)
    TN_test = np.zeros(SAMPLE_SIZE);  FN_test = np.zeros(SAMPLE_SIZE)

    # train/test the model several several times
    for s in range(SAMPLE_SIZE):
        # train the model
        weights = trainModel(X_train, Y_train, N_HIDDEN, LRATE, NUM_EPOCHS[i], 0)
        # evalute on the training set
        TP_train[s], TN_train[s], FP_train[s], FN_train[s], _ = testModel(X_train, Y_train, weights, 0)
        # evalute on the test set
        TP_test[s], TN_test[s], FP_test[s], FN_test[s], _ = testModel(X_test, Y_test, weights, 0)

    # compute the mean
    TP_train_mean = np.mean(TP_train); TN_train_mean = np.mean(TN_train)
    FP_train_mean = np.mean(FP_train); FN_train_mean = np.mean(FN_train)
    TP_test_mean = np.mean(TP_test); TN_test_mean = np.mean(TN_test)
    FP_test_mean = np.mean(FP_test); FN_test_mean = np.mean(FN_test)
    # compute the error
    err_train[i] = 1.0 - computeAccuracy(TP_train_mean, TN_train_mean, FP_train_mean, FN_train_mean)
    err_test[i] = 1.0 - computeAccuracy(TP_test_mean, TN_test_mean, FP_test_mean, FN_test_mean)

# plot the training versus test set error
plt.figure(1)
LW = 2.0
plt.plot(NUM_EPOCHS, err_train, marker='o', linewidth=LW)
plt.plot(NUM_EPOCHS, err_test, marker='x', linewidth=LW)
plt.legend(['training set error', 'test set error'], loc='upper right')
plt.title('Number of hidden units = %d\n Data = %s' % (N_HIDDEN, fname_test))
plt.ylabel('Error'); plt.xlabel('Training Epochs')
plt.show()