import matplotlib.pyplot as plt
import numpy as np
from nn_alg import *
from sklearn import metrics

fname_train = 'heart_train.arff.txt'
fname_test  = 'heart_test.arff.txt'
# fname_train = 'lymph_train.arff.txt'
# fname_test  = 'lymph_test.arff.txt'

LRATE = .1
N_HIDDEN = 5
NUM_EPOCHS = 500

# load data
X_train, Y_train, feature_mean, feature_std = loadData(fname_train, True)
X_test, Y_test, _,_ = loadData(fname_test, False, feature_mean, feature_std)

# train the model
weights = trainModel(X_train, Y_train, N_HIDDEN, LRATE, NUM_EPOCHS, 1)
# evalute on the test set
TP, TN, FP, FN, Y_hat = testModel(X_test, Y_test, weights, 1)



fpr, tpr, thresholds = metrics.roc_curve(Y_test, Y_hat, pos_label=1)
plt.figure(2)
LW = 2.0
plt.plot(fpr, tpr, marker='x', linewidth=LW)
plt.title('Number of hidden units = %d, Test set Accuracy = %.4f\n Data = %s'
          % (N_HIDDEN, computeAccuracy(TP, TN, FP, FN), fname_test))
plt.ylabel('True Positive Rate');
plt.xlabel('False Positive Rate')
plt.show()