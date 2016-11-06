from util import *
import numpy as np
import scipy.io.arff as sparff
import sys

def loadData(data_):
    # read the training data
    data, metadata = sparff.loadarff(data_)
    # numerical feature standardization
    data = featureNormalization(data, metadata)
    num_features = len(metadata.names()) - 1
    num_instances = len(data)
    feature_info = metadata[metadata.names()[num_features]]
    # convert data to a list of lists
    X, Y = [], []
    for m in range(num_instances):
        # convert labels to 0-1 encoding
        Y.append(feature_info[1].index(data[m][num_features]))
        # create feature vector representation for each isntance
        featureVector = []
        for n in range(num_features):
            this_feature_info = metadata[metadata.names()[n]]
            # continuous valued feature - one value
            if this_feature_info[0].lower() == TYPE_NUMERIC:
                featureVector.append(data[m][n])
            # discrete valued feature - one hot encoding
            elif this_feature_info[0].lower() == TYPE_NOMINAL:
                placeholder = np.zeros(len(this_feature_info[1]),)
                placeholder[this_feature_info[1].index(data[m][n])] = 1
                featureVector.extend(placeholder)
            else:
                raise ValueError('Unrecognizable feature type.\n')
        featureVector.append(1)
        X.append(featureVector)

    return X, Y


def featureNormalization(data, metadata):
    num_features = len(metadata.names()) - 1
    num_instances = len(data)
    # loop over all features
    for n in range(num_features):
        # find numerical features
        feature_info = metadata[metadata.names()[n]]
        if feature_info[0] == TYPE_NUMERIC:
            vals = np.zeros(num_instances,)
            # loop over all instances to compute mean and std
            for m in range(num_instances):
                vals[m] = data[m][n]
            feature_mean = np.mean(vals)
            feature_std = np.std(vals)
            # loop over all instances to do normalization
            for m in range(num_instances):
                data[m][n] = 1.0 * (data[m][n] - feature_mean) / feature_std
    return data

def initWeights(inputDim, nHidden):
    weights = []
    if nHidden == 0:
        weights.append(np.random.uniform(WEIGHTS_INIT_LB, WEIGHTS_INIT_UB, inputDim))
    elif nHidden > 0:
        weights.append(np.random.uniform(WEIGHTS_INIT_LB, WEIGHTS_INIT_UB, (nHidden, inputDim)))
        weights.append(np.random.uniform(WEIGHTS_INIT_LB, WEIGHTS_INIT_UB, nHidden))
    else:
        raise ValueError('Number of hidden units need to be a positive integer.\n')
    return weights


def sigmoid(input):
    return np.divide(1.0,(np.add(1.0,np.exp(-input))))


def loadSimpleData(mapping_type):
    X = np.reshape(np.array([0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1]), (4,3))
    if mapping_type is 'or':
        Y = np.array([0, 1, 1, 1])
    elif mapping_type is 'and':
        Y = np.array([0, 0, 0, 1])
    elif mapping_type is 'xor':
        Y = np.array([0, 1, 1, 0])
    else:
        raise ValueError('Unrecognizable pattern type.\n')
    return X, Y


def nn_predict(wts, rawInput):
    if len(wts) == 1:
        output = sigmoid(np.dot(rawInput, wts[0]))
    elif len(wts) == 2:
        hiddenAct = sigmoid(np.matmul(wts[0], rawInput))
        output = sigmoid(np.dot(hiddenAct, wts[1]))
    else:
        raise ValueError('Unrecognizable weight cardinality.\n')
    return output


def deltaLearn(X, Y, wts, lrate):
    # one sweep through the entire training set
    orderedIdx = np.random.permutation(len(Y))
    error, counts, mae = 0,0,0

    for m in orderedIdx:
        # forward prop
        rawInput = np.array(X[m])
        output = sigmoid(np.dot(rawInput, wts[0]))
        delta = Y[m] - output
        # accumulate gradient
        wts_gradient = np.multiply(lrate * delta, rawInput)
        # update weights
        wts[0] += wts_gradient
        # record performance measure
        mae += np.abs(delta)
        error += crossEntropyError(Y[m], output)
        if (output > THRESHOLD and Y[m] == 1) or (output < THRESHOLD and Y[m] == 0):
            counts += 1
    return wts, error, counts, mae


def backprop(X, Y, wts, lrate):
    # preallocate for gradient accumulation
    wts_gradient = []
    wts_gradient.append(np.zeros(np.shape(wts[0])))
    wts_gradient.append(np.zeros(np.shape(wts[1])))
    error, counts, mae = 0,0,0

    # one sweep through the entire training set
    orderedIdx = np.random.permutation(len(Y))
    for m in orderedIdx:
        # forward prop
        rawInput = np.array(X[m])
        hiddenAct = sigmoid(np.matmul(wts[0], rawInput))
        output = sigmoid(np.dot(hiddenAct, wts[1]))
        # backprop
        delta_o = Y[m] - output
        delta_h = delta_o * wts[1] * hiddenAct * (1 - hiddenAct)
        # compute gradient
        wts_gradient[1] = delta_o * hiddenAct
        wts_gradient[0] = np.outer(delta_h, rawInput)
        # update weights
        wts[1] += lrate * wts_gradient[1]
        wts[0] += lrate * wts_gradient[0]
        # record performance measure
        mae += np.abs(delta_o)
        error += crossEntropyError(Y[m], output)
        if (output > THRESHOLD and Y[m] == 1) or (output < THRESHOLD and Y[m] == 0):
            counts += 1
    return wts, error, counts, mae


def crossEntropyError(y, y_hat):
    # TODO: handle log(0)
    return -y * np.log(y_hat+SMALL_NUM) - (1-y) * np.log(1-y_hat+SMALL_NUM)


def trainModel(X_train, Y_train, nHidden, lrate, nEpochs, printOutput = True):
    # initialize the weights to uniform random values
    wts = initWeights(len(X_train[0]), nHidden)

    # train the model for some number of epochs
    for e in range(nEpochs):
        # update weights w.r.t one sweep of the training data
        # model without hidden units
        if nHidden == 0:
            wts, error, counts, mae = deltaLearn(X_train, Y_train, wts, lrate)
        # general multilayerd model
        elif nHidden > 0:
            wts, error, counts, mae = backprop(X_train, Y_train, wts, lrate)
        else:
            raise ValueError('Number of hidden units need to be postiive.\n')

        if (np.mod(e, 100) == 0) and printOutput:
            print ('Trainging Epoch = %6.d, CEE = %6.4f, MAE = %6.4f, nRight = %d, nWrong = %d'
                   % (e, error, mae, counts, len(Y_train) - counts))
    return wts


def testModel(X_test, Y_test, wts, printOutput = True):
    truePostive, trueNegative, falsePositive, falseNegative = 0, 0, 0, 0
    outputs = []
    for m in range(len(Y_test)):
        # forward prop
        output = nn_predict(wts, X_test[m])
        if Y_test[m] == 1:
            if output > THRESHOLD:
                truePostive +=1
            else:
                falseNegative +=1
        elif Y_test[m] == 0:
            if output < THRESHOLD:
                trueNegative +=1
            else:
                falsePositive +=1
        else:
            raise ValueError('Y is neither 1 or 0.\n')


        outputs.append(output)
        if printOutput: # print a prediction vs. target for each instance
            print ('Prediction = %.3f | Target = %.3f' % (output, Y_test[m]))
    if (truePostive + trueNegative + falsePositive + falseNegative) != len(Y_test):
        raise ValueError('TP + FP + TN + FN != |Y|.\n')

    if printOutput: # print overall performance
        print ('Test Performance = %.3f (Baseline = %.3f)' %
               (computeAccuracy(truePostive, trueNegative, falsePositive, falseNegative),
                countBaseRate(Y_test)))

    return truePostive, trueNegative, falsePositive, falseNegative, outputs


def computeAccuracy(truePostive, trueNegative, falsePositive, falseNegative):
    return 1.0 * (truePostive + trueNegative) / (truePostive + trueNegative + falsePositive + falseNegative)