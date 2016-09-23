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


def getFeatureVals(data_, metadata_):
    feature_vals_unique = []
    columns = []
    # read each instance
    for i in range(len(metadata_.types())):
        column = []
        # append the ith feature value of the current instance
        for instance in data_:
            column.append(instance[i])
        columns.append(column)

        # save sorted unique values for each feature
        feature_vals_unique.append(sorted(list(set(column))))
    return feature_vals_unique, columns



def loadData(data_):
    # read the training data
    data, metadata = sp.loadarff(data_)

    # from metadata, save feature valus as a list of tuples
    feature_range = []
    for name in metadata.names():
        feature_range.append(metadata[name][1])

    # read the possible values for each feature
    feature_vals_unique, columns = getFeatureVals(data, metadata)

    return data, metadata, feature_range, feature_vals_unique, columns



###################### DT Functions ##########################


def splitData_continuous(data_, featureIdx_, threshold_):
    '''

    :param data_:
    :param featureIdx_:
    :param threshold_: should be mean(v_i, v_i+1) for some i
    :return:
    '''
    data_divided = []
    data_sub_less = []
    data_sub_greater = []
    # loop over all instances
    for instance in data_:
        # assign to corresponding subset based on a comparison w/ a threshold
        if instance[featureIdx_] <= threshold_:
            data_sub_less.append(instance)
        else:
            data_sub_greater.append(instance)

    data_divided.append(data_sub_less)
    data_divided.append(data_sub_greater)
    return data_divided



def splitData_discrete(data_, featureIdx_, feature_vals_):
    '''
    Divided the data set with respect to feature F
    :param data_:
    :param featureIdx_:
    :param feature_vals_:
    :return:
    '''
    data_divided = []
    # for ith feature value, F_i
    for _feature_val in feature_vals_:
        data_sub = []
        # loop over all instances
        for instance in data_:
            # collect instances with F_i
            if instance[featureIdx_] == _feature_val:
                data_sub.append(instance)
        # save the subset
        data_divided.append(data_sub)
    return data_divided


def neighbourMean(nparray):
    '''
    Find all possible threshold values given the a set of ordered feature values
    :param nparray: order continuous valued feature vector
    :return: threshold
    '''
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

    # if the class is pure, then there is no entropy
    if count == len(label_col) or count == 0:
        return 0
    # compute the entropy for non-pure sample
    else:
        # MLE for p
        p = 1.0 * count / len(label_col)
        # compute entropy
        entropy = -p * np.log2(p) - (1-p) * np.log2(1-p)
        return entropy


def computeEntropy_dividedSet(data_divded, data_whole, classRange_):
    entropy = 0
    # for each subset
    for data_sub in data_divded:
        # collect all class labels
        classLabels = []
        for instance in data_sub:
            classLabels.append(instance[-1])

        # weight the entropy by the occurence
        p_x = 1.0 * len(data_sub) / len(data_whole)
        entropy += p_x * computeEntropy_binary(classLabels, classRange_)
    return entropy


def findBestThreshold(data_train, featureIdx, feature_vals, yRange_):
    allThresholds = neighbourMean(np.array(feature_vals[featureIdx]))
    # find the threshold (split) with the lowest entropy
    bestThreshold = allThresholds[0]
    minEntropy = float('inf')
    for t in range(len(allThresholds)):
        # split the data with the t-th threshold
        data_divded = splitData_continuous(data_train, featureIdx, allThresholds[t])
        entropy_temp = computeEntropy_dividedSet(data_divded, data_train, yRange_)
        # keep track of the min entropy and its index
        # when there is a tie, pick the first one, achieved by < (instead of <=)
        if entropy_temp < minEntropy:
            minEntropy = entropy_temp
            bestThreshold = allThresholds[t]
    return minEntropy, bestThreshold



def computeConditionalEntropy(data_, feature_vals_, metadata_, yRange_, feature_used_):
    # loop over all features
    nFeatures = len(metadata_.types())
    entropy_YgX = np.zeros((nFeatures-1,))
    entropy_YgX.fill(np.NaN)

    for i in range(nFeatures - 1):
        # skip features that is alread used
        if feature_used_[i]: continue
        # condition 1: numeric feature
        if isNumeric(metadata_.types()[i]):
            entropy, bestThreshold_idx = findBestThreshold(data_, i, feature_vals_, yRange_)

        # condition 2: nominal feature
        else:
            # split the data with the ith feature
            data_divded = splitData_discrete(data_, i, feature_vals_[i])
            # accumulate entropy
            entropy = computeEntropy_dividedSet(data_divded, data_, yRange_)

        # save the entropy
        entropy_YgX[i] = entropy
    return entropy_YgX


def computInfoGain(yLabels_, yRange_, data_, feature_vals_, metadata_, feature_used_):
    # compute information gain for all features
    entropy_Y = computeEntropy_binary(yLabels_, yRange_)
    entropy_YgX = computeConditionalEntropy(data_, feature_vals_, metadata_, yRange_, feature_used_)
    infomationGain = np.subtract(entropy_Y, entropy_YgX)
    return infomationGain


###################### TREE FUNCTIONS ##########################

def setIsPure(data_):
    labels = []
    for instance in data_:
        labels.append(instance[-1])
    labels = set(labels)

    if len(labels) == 1:
        return True
    return False


def stoppingGrowing(data_):

    # (i) all of the training instances reaching the node belong to the same class
    if setIsPure(data_):
        return True
    # (ii) number of training instances reaching the node < m
    if len(data_) < m:
        return True
    # (iii) no feature has positive information gain


    # (iv) there are no more remaining candidate splits at the node.
    return False



def findBestSplit(classLabels_, classLabelsRange_, data_train_, feature_vals_unique_,
                  metadata_, feature_used_):
    '''
    Return feature that maximize information gain
    :param classLabels_:
    :param classLabelsRange_:
    :param data_train_:
    :param feature_vals_unique_:
    :param metadata_:
    :return:
    '''
    infomationGain = computInfoGain(classLabels_, classLabelsRange_, data_train_,
                                    feature_vals_unique_, metadata_, feature_used_)
    # select the best feature
    # Note that feature used will have NAN as its infoGain
    best_feature_index = np.nanargmax(infomationGain)

    # TODO delete print
    print infomationGain
    return best_feature_index



def recordUsedFeature(feature_used_, best_feature_idx_):
    # if the input feature is discrete, then we can nolonger split on it
    if not isNumeric(metadata.types()[best_feature_idx_]):
        feature_used_[best_feature_idx_] = True
    # else:
        # TODO check if numeric, we should be able to split further (is this true?)
    return feature_used_


def getMajorityClass(data_, classlabel_range_):
    '''
    get the majority vote for the current input data set
    if there is a tie, choose the first class listed in "classlabel_range_"
    the tie is handled by np.argmax()
    :param data_: the current data set
    :param classlabel_range_: all possible values for the class, should be binary
    :return: the majority vote
    '''
    if not len(classlabel_range_) == 2:
        raise Exception('ERROR:non-binary class detected')
    count = np.zeros(2,)
    # loop over all training example, count the occurence of each class value
    for instance in data_:
        if instance[-1] == classlabel_range_[0]:
            count[0] +=1
        elif instance[-1] == classlabel_range_[1]:
            count[1] +=1
        else:
            raise Exception('ERROR:un-recognizable class label for the training example')
    # return the majority
    majorityClassLabel = classlabel_range_[np.argmax(count)]
    return majorityClassLabel



def initTreeNode(data_, metadata_, classLabelsRange_, best_feature_idx_, feature_used_,
                 feature_val_, parent_):
    '''
    Initialize a tree node
    Note: this method doesn't initialize the children list
    :param data_:
    :param metadata_:
    :param classLabelsRange_:
    :param best_feature_idx_:
    :param feature_used_:
    :param feature_val_:
    :param parent_:
    :return:
    '''
    # create the root
    n_feature_name = metadata_.names()[best_feature_idx_]
    n_feature_type = metadata_.types()[best_feature_idx_]
    n_feature_val = feature_val_
    n_feature_used = feature_used_
    n_parent = parent_
    # n_children = children_
    n_label = getMajorityClass(data_, classLabelsRange_)
    # make the node
    node = decisionTreeNode()
    node.setFeature(n_feature_name, n_feature_type, n_feature_val, n_feature_used)
    node.setClassificationLabel(n_label)
    node.setParent(n_parent)
    # node.setChildren(n_children)
    return node

def initLeaf(data_, classLabelsRange_, feature_used_, feature_val_, parent_):
    '''

    :param data_:
    :param classLabelsRange_:
    :param feature_used_:
    :param feature_val_:
    :param parent_:
    :return:
    '''
    # create the root
    n_feature_name = None
    n_feature_type = None
    n_feature_val = feature_val_
    n_feature_used = feature_used_
    n_parent = parent_
    # n_children = children_
    n_label = getMajorityClass(data_, classLabelsRange_)
    # make the node
    node = decisionTreeNode()
    node.setFeature(n_feature_name, n_feature_type, n_feature_val, n_feature_used)
    node.setClassificationLabel(n_label)
    node.setParent(n_parent)
    return node


def makeSubtree(data_, metadata_, classLabels_, classLabelsRange_, feature_range_,
                feature_vals_unique_, feature_used_, feature_val_cur_, parent_):

    if stoppingGrowing(data_):
        # make a leaf node N
        leaf = initLeaf(data_, classLabelsRange_, feature_used_, feature_val_cur_, parent_)
        return leaf

    else:
        # pick the best feature
        best_feature_idx = findBestSplit(classLabels_, classLabelsRange_, data_,
                                         feature_vals_unique_, metadata_, feature_used_)
        # update feature selection indicator
        feature_used_ = recordUsedFeature(feature_used_, best_feature_idx)

        # start creating the tree
        node = initTreeNode(data_, metadata_, classLabelsRange_, best_feature_idx, feature_used_,
                            feature_val_cur_, parent_)

        # split the data using the best feature
        # handle continous and discrete feature separately
        if isNumeric(metadata_.types()[best_feature_idx]):
            _, bestThreshold = findBestThreshold(data_, best_feature_idx, feature_vals_unique_,
                                                 classLabelsRange_)
            data_divided = splitData_continuous(data_, best_feature_idx, bestThreshold)
        else:
            data_divided = splitData_discrete(data_, best_feature_idx, feature_vals_unique_)


        # for eachOutcome k of S, create children
        for i in range(len(feature_range_[best_feature_idx])):
            # Dk = subset of instances that have outcome k
            feature_value_cur_ = feature_range_[best_feature_idx][i]
            data_cur = data_divided[i]
            parent_ = node
            child_i = makeSubtree(data_cur, metadata_, classLabels_, classLabelsRange_,
                                  feature_range_, feature_vals_unique_, feature_used_,
                                  feature_value_cur_, parent_)
            node.setChildren(child_i)
    # return subtree rooted at N
    return node

###################### END OF DEFINITIONS OF HELPER FUNCTIONS ##########################

# read input arguments
fname_train, fname_test, m = processInputArgs()
# load data
# feature_range [tuple] = the possible values for discrete feature, None for continous
# feature_vals_unique [list] = the observed values for all features
# columns [list] = the actual feature vector from data_train
data_train, metadata, feature_range, feature_vals_unique, columns = loadData(fname_train)

# read some parameters
feature_used = np.zeros((len(metadata.types()) - 1,), dtype=bool)
classLabels = columns[-1]
classLabelsRange = feature_range[-1]

# test
printAllFeatures(metadata, feature_range)
print "\n"

feature_val_cur = None
parent = None
root = makeSubtree(data_train, metadata, classLabels, classLabelsRange, feature_range,
            feature_vals_unique, feature_used, feature_val_cur, parent)

