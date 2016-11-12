from nn_alg import *
# load data
lrate, nHidden, nEpochs, fname_train, fname_test = processInputArgs_nnet()
X_train, Y_train, feature_mean, feature_std = loadData(fname_train, True)
X_test, Y_test, _,_ = loadData(fname_test, False, feature_mean, feature_std)

# train the model
weights = trainModel(X_train, Y_train, nHidden, lrate, nEpochs)
# evalute on the test set
TP, TN, FP, FN, _ = testModel(X_test, Y_test, weights)