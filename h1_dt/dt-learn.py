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


def getFeatureVals(data, metadata):
    feature_vals = []
    # read each instance
    for i in range(len(metadata.types())):
        vals = []
        for instance in data:
            # append the ith feature value of the current instance
            vals.append(instance[i])
            if isContinuous(metadata.types()[i]):
                # sort continuous valued list
                vals = sorted(vals)
            else:
                # find unique values for nominal data
                vals = list(set(vals))
        feature_vals.append(vals)
    return feature_vals



def loadData(fname_data):
    # read the training data
    data, metadata = sp.loadarff(fname_data)
    # read the possible values for each feature
    feature_vals = getFeatureVals(data, metadata)
    return data, metadata, feature_vals

# verify: for all features, all feature values IN feature_range (read from the data)
# delete before submit!
def dataChecker(data):
    for instance in data:
        for i in xrange(len(instance)):
            if feature_vals[i] != TYPE_NUMERIC and instance[i] not in feature_vals[i]:
                print "ERROR: Unrecognizable Feature!"
    return 0


def printAllFeatures(metadata, feature_vals):
    for i in range(len(feature_vals)):
        if isContinuous(metadata.types()[i]):
            featurevalues = "[......]"
        else:
            featurevalues = str(feature_vals[i])
        print "%d\t%8s\t%s\t%s " % \
              (i, metadata.names()[i], metadata.types()[i], featurevalues)

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
        # determine class label/probabilities for N
    else:
        # make an internal node N
        S = findBestSplit(D, C)
    #     for eachOutcome k of S
    #         Dk = subset of instances that have outcome k
    #         kth child of N = MakeSubtree(Dk)
    # return subtree rooted at N
    return 0


def computeEntropy_binary(p):
    entropy = -p * np.log(p) - (1 - p) * np.log(1 - p)
    return entropy

def entropy(classLabels, classRange):
    # assume the class range has cardinality 2
    n_total = len(classLabels)
    count = 0
    count_ = 0
    for label in classLabels:
        if label == classRange[0]:
            count+=1
        else:
            count_+=1

    if (count + count_) != n_total:
        sys.exit('ERROR')

    # MLE for p
    p = 1.0 * count / n_total
    # compute entropy
    entropy = computeEntropy_binary(p)
    return entropy

###################### END OF DEFINITIONS OF HELPER FUNCTIONS ##########################

# read input arguments
fname_train, fname_test, m = processInputArgs()
# load data
data_train, metadata, feature_vals = loadData(fname_train)

# read some parameters
nTrain = len(data_train)
nFeature = len(metadata.types())

# test
printAllFeatures(metadata, feature_vals)

# start creating the tree
node = decisionTreeNode()


print feature_vals[-1]


# # read each instance
# for instance in data_train:
#     # read each feature
#     for i in xrange(len(instance)):
#         feature_i = instance[i]
#         print "%s \t %s" % (metadata.types().pop(i), feature_i)
#     sys.exit('STOP')


