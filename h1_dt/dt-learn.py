import numpy as np
import scipy.io.arff as sp
import sys
import arff


from decisionTreeNode import decisionTreeNode
from util import *


def processInputArgs():
    # process input arguments
    if len(sys.argv) == 4:
        fname_train = str(sys.argv[1])
        fname_test = str(sys.argv[2])
        m = int(str(sys.argv[3]))
    else:
        sys.exit('ERROR: This program takes exactly 3 input arguments.')
    return fname_train, fname_test, m


def getFeatureVals(_data, _metadata):
    feature_vals = []
    columns = []
    # read each instance
    for i in range(len(_metadata.types())):
        column = []
        # append the ith feature value of the current instance
        for instance in _data:
            column.append(instance[i])
        columns.append(column)

        # save sorted unique values for each feature
        feature_vals.append(sorted(list(set(column))))
    return feature_vals, columns



def loadData(_data):
    # read the training data
    data, metadata = sp.loadarff(_data)
    # read the possible values for each feature
    feature_vals, columns = getFeatureVals(data, metadata)
    return data, metadata, feature_vals, columns



###################### DT Functions ##########################

def determineCandidateSplits(data):
    return 0

def stoppingGrowing():
    return 0

def findBestSplit():
    return 0

def makeSubtree(data):
    C = DetermineCandidateSplits(D)
    if stoppingGrowing():
        temp = 0
        # make a leaf node N
        # determine class label for N
    else:
        # make an internal node N
        S = findBestSplit(D, C)
    #     for eachOutcome k of S
    #         Dk = subset of instances that have outcome k
    #         kth child of N = MakeSubtree(Dk)
    # return subtree rooted at N
    return 0



def splitData_continuous(_data, _featureIdx, _threshold):
    '''

    :param _data:
    :param _featureIdx:
    :param _threshold: should be mean(v_i, v_i+1) for some i
    :return:
    '''
    data_divided = []
    data_sub_less = []
    data_sub_greater = []
    # loop over all instances
    for instance in _data:
        # assign to corresponding subset based on a comparison w/ a threshold
        if instance[_featureIdx] <= _threshold:
            data_sub_less.append(instance)
        else:
            data_sub_greater.append(instance)

    data_divided.append(data_sub_less)
    data_divided.append(data_sub_greater)
    return data_divided



def splitData_discrete(_data, _featureIdx, _feature_vals):
    '''
    Divided the data set with respect to feature F
    :param _data:
    :param _featureIdx:
    :param _feature_vals:
    :return:
    '''
    data_divided = []
    # for ith feature value, F_i
    for _feature_val in _feature_vals:
        data_sub = []
        # loop over all instances
        for instance in _data:
            # collect instances with F_i
            if instance[_featureIdx] == _feature_val:
                data_sub.append(instance)
        # save the subset
        data_divided.append(data_sub)
    return data_divided


def neighbourMean(nparray):
    return np.divide(np.add(nparray[0:-1], nparray[1:]), 2.0)


def computeEntropy_binary(label_col, label_range):
    # assume the class range has cardinality 2
    if not len(label_range) == 2:
        sys.exit('ERROR: non-binary class labels.')

    # count the frequency for one class
    count = 0
    for label in label_col:
        if label == label_range[0]:
            count+=1

    if count == len(label_col) or count == 0:
        return 0
    else:
        # MLE for p
        p = 1.0 * count / len(label_col)
        # compute entropy
        entropy = -p * np.log2(p) - (1-p) * np.log2(1-p)
        return entropy


def computeEntropy_dividedSet(_data_divded, _data_whole, _classRange):
    entropy = 0
    for data_sub in _data_divded:
        classLabels = []
        for instance in data_sub:
            classLabels.append(instance[-1])

        # weight the entropy by the occurence
        p_x = 1.0 * len(data_sub) / len(_data_whole)
        entropy += p_x * computeEntropy_binary(classLabels, _classRange)
    return entropy


def findBestThreshold(data_train, featureIdx, feature_vals):
    allThresholds = neighbourMean(np.array(feature_vals[featureIdx]))
    # find the threshold (split) with the lowest entropy
    bestThreshold = allThresholds[0]
    minEntropy = float('inf')
    for t in range(len(allThresholds)):
        # split the data with the t-th threshold
        data_divded = splitData_continuous(data_train, featureIdx, allThresholds[t])
        entropy = computeEntropy_dividedSet(data_divded, data_train, feature_vals[-1])
        # keep track of the min entropy and its index
        if entropy < minEntropy:
            minEntropy = entropy
            bestThreshold = allThresholds[t]
    return minEntropy, bestThreshold



def computeConditionalEntropy():
    # loop over all features
    entropy_YgX = np.zeros((nFeature - 1, 1,))

    for i in range(nFeature - 1):
        # condition 1: numeric feature
        if isNumeric(metadata.types()[i]):
            entropy, bestThreshold_idx = findBestThreshold(data_train, i, feature_vals)

        # condition 2: nominal feature
        else:
            # split the data with the ith feature
            data_divded = splitData_discrete(data_train, i, feature_vals[i])
            # accumulate entropy
            entropy = computeEntropy_dividedSet(data_divded, data_train, feature_vals[-1])

        entropy_YgX[i] = entropy

    return entropy_YgX

###################### END OF DEFINITIONS OF HELPER FUNCTIONS ##########################

# read input arguments
fname_train, fname_test, m = processInputArgs()
# load data
data_train, metadata, feature_vals, columns = loadData(fname_train)

# read some parameters
nTrain = len(data_train)
nFeature = len(metadata.types())

# test
printAllFeatures(metadata, feature_vals)
print "\n"

entropy_Y = computeEntropy_binary(columns[-1], feature_vals[-1])
entropy_YgX = computeConditionalEntropy()
infomationGain = np.subtract(entropy_Y, entropy_YgX)

print infomationGain


# start creating the tree
# node = decisionTreeNode()


# entropy_YgX = []
# for i in range(nFeature-1):
#     # split the training examples according to X_i
#     temp = 0
#     # compute the resulting entropy for X_i
#     # compute the info gain for X_i
#
# # find X_i that maximize info gain
#
# # split on X_i