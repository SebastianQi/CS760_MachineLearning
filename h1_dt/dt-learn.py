import numpy as np
import scipy.io.arff as sp
import sys

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


def getFeatureRange(fname_train):
    feature_range = []
    feature_idx = 0
    for line in open(fname_train):
        if line.strip().startswith(ATTRIBUTE_INDICATOR):
            # read the line
            range_str = line.rstrip().split(' ')
            # pop out line header and feature name
            range_str.pop(0);range_str.pop(0)
            # check if the feature is continuous or discrete
            if isContinuous(range_str[0]):
                feature_range.append(TYPE_NUMERIC)
            else:
                # remove the two brakets
                range_str.pop(0)
                range_str[-1] = range_str[-1][:-1]
                # remove commas
                for i in range(len(range_str)):
                    range_str[i] = range_str[i].strip(',')
                # append the feature values
                feature_range.append(range_str)
        feature_idx +=1
    return feature_range

def loadData(fname_data):
    # read the training data
    data, metadata = sp.loadarff(fname_data)
    # read the possible values for each feature
    feature_vals = getFeatureRange(fname_data)
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
        print "%d\t%8s\t%s\t%s " % \
              (i, metadata.names()[i], metadata.types()[i], str(feature_vals[i]))

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




###################### END OF DEFINITIONS OF HELPER FUNCTIONS ##########################

# read input arguments
fname_train, fname_test, m = processInputArgs()
# load data
data_train, metadata, feature_vals = loadData(fname_train)
# data_test, _, _ = loadData(fname_test)
# nTest = len(data_test)

# read some parameters
nTrain = len(data_train)
nFeature = len(metadata.types())

node = decisionTreeNode()

printAllFeatures(metadata, feature_vals)



# # read each instance
# for instance in data_train:
#     # read each feature
#     for i in xrange(len(instance)):
#         feature_i = instance[i]
#         print "%s \t %s" % (metadata.types().pop(i), feature_i)
#
#     sys.exit('STOP')


