import matplotlib.pyplot as plt
from kNN_alg import *
from util import *


# read input arguments
[fname_train, fname_test] = ['yeast_train.arff.txt', 'yeast_test.arff.txt']

# load data
X_train, Y_train, metadata = loadData(fname_train)
X_test, Y_test, _ = loadData(fname_test)
Y_range = metadata[metadata.names()[-1]][1]

Ks = [1, 5, 10, 20, 30]
accuracy_test = np.zeros(len(Ks),)
for i in range(len(Ks)):
    accuracy_test[i], _ = testModel(X_train, Y_train, X_test, Y_test, Y_range, Ks[i], False)

plt.figure(1)
plt.plot(Ks, accuracy_test)

plt.title('%s' % fname_test)
plt.ylabel('Test set classification accuracy')
plt.xlabel('K')
plt.show()

