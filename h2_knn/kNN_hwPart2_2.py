import matplotlib.pyplot as plt
from kNN_alg import *
from util import *


# read input arguments
[fname_train, fname_test] = ['wine_train.arff.txt', 'wine_test.arff.txt']

# load data
X_train, Y_train, metadata = loadData(fname_train)
X_test, Y_test, _ = loadData(fname_test)
Y_range = metadata[metadata.names()[-1]][1]

Ks = [1, 2, 3, 5, 10]
accuracies_test = []
for i in range(len(Ks)):
    accuracy = testModel(X_train, Y_train, X_test, Y_test, Y_range, Ks[i], False)
    accuracies_test.append(accuracy)


plt.figure(1)
plt.plot(Ks, accuracies_test)

plt.title('%s' % fname_test)
plt.ylabel('Test set mean absolute error')
plt.xlabel('K')
plt.show()

