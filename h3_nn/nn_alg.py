from util import *

import numpy as np
import scipy.io.arff as sparff
import sys

def loadData(data_):
    '''
    The network uses a single input unit to represent each numeric feature,
    and a one-of-k encoding for each nominal feature.
    :param data_:
    :return:
    '''
    # read the training data
    data, metadata = sparff.loadarff(data_)
    M = len(metadata.names()) - 1
    N_features = len(data)
    label_info = metadata[metadata.names()[M]]
    # convert data to a list of lists
    X = []
    Y = []
    for n in range(N_features):
        # convert labels to 0-1 encoding
        Y.append(label_info[1].index(data[n][M]))

        # create feature vector representation for each isntance
        featureVector = []
        for m in range(M):
            this_feature_info = metadata[metadata.names()[m]]
            # continuous valued feature - one value
            if this_feature_info[0].lower() == TYPE_NUMERIC:
                featureVector.append(data[n][m])
            # discrete valued feature - one hot encoding
            elif this_feature_info[0].lower() == TYPE_NOMINAL:
                placeholder = np.zeros(len(this_feature_info[1]),)
                placeholder[this_feature_info[1].index(data[n][m])] = 1
                featureVector.extend(placeholder)
            else:
                raise ValueError('Unrecognizable feature type.\n')
        featureVector.append(1)
        X.append(featureVector)

    return X, Y, metadata


def processMetadata(metadata):
    # process metadataS
    inputDim = 0
    for name in metadata.names():
        this_feature_info = metadata[name]
        if this_feature_info[0].lower() == TYPE_NUMERIC:
            inputDim += 1
        elif this_feature_info[0].lower() == TYPE_NOMINAL:
            inputDim += len(this_feature_info[1])
        else:
            raise ValueError('Unrecognizable feature type.\n')

    inputDim = inputDim - 2 + 1 # assume y is binary & add 1 for the bias term
    return inputDim


def initWeights(inputDim, nHidden):
    '''
    1. If h = 0, the network should have no hidden units, and the input units should be directly
    connected to the output unit. Otherwise, if h > 0, the network should have a single
    layer of h hidden units with each fully connected to the input units and the output unit.
    2. All weights and bias parameters are initialized to random values in [-0.01, 0.01].
    3. Your network is intended for binary classification problems, and therefore it has one
    output unit with a sigmoid function.
    :param inputDim: feature vector dimension
    :param nHidden: number of hidden units
    :return: a weight vector if no hidden units; 2 matrices if hiddenUnits > 0
    '''
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



def testModel(X_test, Y_test, wts):
    counts = 0
    for m in range(len(Y_test)):
        # forward prop
        output = nn_predict(wts, X_test[m])
        print ('Prediction = %.3f | Target = %.3f' %(output, Y_test[m]))
        if output > .5 and Y_test[m] == 1:
            counts += 1
        elif output < .5 and Y_test[m] == 0:
            counts += 1

    print ('Test Performance = %.3f (Baseline = %.3f)' %
           (1.0 * counts / len(Y_test), countBaseRate(Y_test)))


def nn_predict(wts, rawInput):
    if len(wts) == 1:
        output = sigmoid(np.dot(rawInput, wts[0]))
    elif len(wts) == 2:
        hiddenAct = sigmoid(np.matmul(wts[0], rawInput))
        output = sigmoid(np.dot(hiddenAct, wts[1]))
    else:
        raise ValueError('Unrecognizable weight cardinality.\n')
    return output



def deltaLearn(X_train, Y_train, wts, lrate):
    inputDim = len(X_train[0])
    # one sweep through the entire training set
    for m in range(len(Y_train)):
        # forward prop
        rawInput = np.array(X_train[m])
        output = sigmoid(np.dot(rawInput, wts[0]))
        delta = Y_train[m] - output
        # accumulate gradient
        wts_gradient = np.multiply(lrate * delta, rawInput)
        # update weights
        wts[0] += wts_gradient
    return wts


def backprop(X_train, Y_train, wts, lrate):
    # preallocate for gradient accumulation
    wts_gradient = []
    wts_gradient.append(np.zeros(np.shape(wts[0])))
    wts_gradient.append(np.zeros(np.shape(wts[1])))
    error = 0
    # one sweep through the entire training set
    for m in range(len(Y_train)):
        # forward prop
        rawInput = np.array(X_train[m])
        hiddenAct = sigmoid(np.matmul(wts[0], rawInput))
        output = sigmoid(np.dot(hiddenAct, wts[1]))
        # backprop
        delta_o = Y_train[m] - output
        delta_h = delta_o * wts[1] * hiddenAct * (1 - hiddenAct)
        # compute gradient
        wts_gradient[1] = delta_o * hiddenAct
        wts_gradient[0] = np.outer(delta_h, rawInput)
        # update weights
        wts[1] += lrate * wts_gradient[1]
        wts[0] += lrate * wts_gradient[0]
        error += np.abs(delta_o)

    return wts, error
