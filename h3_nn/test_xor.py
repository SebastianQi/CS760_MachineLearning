from nn_alg import * 
X_train, Y_train = loadSimpleData('xor')
trainModel(X_train, Y_train,2,.333,30000)

