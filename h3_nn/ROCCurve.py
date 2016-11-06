import matplotlib.pyplot as plt
import numpy as np
from nn_alg import *
from sklearn import metrics

# fname_train = 'heart_train.arff.txt'
# fname_test  = 'heart_test.arff.txt'
fname_train = 'lymph_train.arff.txt'
fname_test  = 'lymph_test.arff.txt'

# TODO the parameters are unclear
LRATE = .05
N_HIDDEN = 20
NUM_EPOCHS = 1000

# load data
X_train, Y_train = loadData(fname_train)
X_test, Y_test   = loadData(fname_test)

# train the model
weights = trainModel(X_train, Y_train, N_HIDDEN, LRATE, NUM_EPOCHS, 1)
# evalute on the test set
TP, TN, FP, FN, Y_hat = testModel(X_test, Y_test, weights, 1)




def plotROC(target, confidence):
    # sort (target, prediction) pair
    targ_pred = sorted(zip(target, confidence), key=lambda x: x[1])
    targ_pred = np.array(targ_pred)
    target = targ_pred[:, 0]
    confidence = targ_pred[:, 1]
    # pre allocate
    num_pos = np.sum(target == 1)
    num_neg = np.sum(target == 0)
    TP, FP, last_TP = 0, 0, 0
    TPRs,FPRs = [],[]

    for i in range(1,len(target)):
        if (confidence[i] != confidence[i-1]) and (target[i] == 0) and (TP > last_TP):
            FPR = 1.0 * FP / num_neg
            TPR = 1.0 * TP / num_pos
            FPRs.append(FPR)
            TPRs.append(TPR)
            last_TP = TP

        if target[i] == 1:
            TP +=1
        else:
            FP +=1

    FPR = 1.0 * FP / num_neg
    TPR = 1.0 * TP / num_pos
    FPRs.append(FPR)
    TPRs.append(TPR)

    # plot ROC curve
    plt.figure(1)
    LW = 2.0
    plt.plot(FPRs, TPRs, marker='x', linewidth=LW)
    plt.title('%s' % fname_test)
    plt.ylabel('True Positive Rate'); plt.xlabel('False Positive Rate')
    plt.show()

plotROC(Y_test, Y_hat)



fpr, tpr, thresholds = metrics.roc_curve(Y_test, Y_hat, pos_label=1)
plt.figure(2)
LW = 2.0
plt.plot(fpr, tpr, marker='x', linewidth=LW)
plt.title('%s' % fname_test)
plt.ylabel('True Positive Rate');
plt.xlabel('False Positive Rate')
plt.show()