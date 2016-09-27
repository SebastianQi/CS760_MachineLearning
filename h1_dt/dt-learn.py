import numpy as np
import scipy.io.arff as sparff
import scipy as sp
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


def getFeatureVals(data_,  metadata):
    feature_vals_unique = []
    columns = []
    # read each instance
    for i in range(len(metadata.types())):
        column = []
        # append the ith feature value of the current instance
        for instance in data_:
            column.append(instance[i])
        columns.append(column)

        # save sorted unique values for each feature
        feature_vals_unique.append(sorted(list(set(column))))
    return feature_vals_unique



def loadData(data_):
    # read the training data
    data, metadata = sparff.loadarff(data_)

    # from metadata, save feature valus as a list of tuples
    feature_range = []
    for name in metadata.names():
        feature_range.append(metadata[name][1])

    # read the possible values for each feature
    feature_vals_unique = getFeatureVals(data, metadata)

    return data, metadata, feature_range, feature_vals_unique



###################### DT Functions ##########################


def splitData_continuous(data_, featureIdx_, threshold_):
    data_sub_less = []
    data_sub_greater = []
    # loop over all instances
    for instance in data_:
        # assign to corresponding subset based on a comparison w/ a threshold
        if instance[featureIdx_] <= threshold_:
            data_sub_less.append(instance)
        else:
            data_sub_greater.append(instance)
    return [data_sub_less, data_sub_greater]


def splitData_discrete(data_, featureIdx_, this_feature_range_):
    '''
    Divided the data set with respect to feature F
    '''
    data_divided = []
    # for ith feature value, F_i
    for _feature_val in this_feature_range_:
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
    :return: thresholds
    '''
    nparray = np.sort(np.unique(nparray))
    return np.divide(np.add(nparray[0:-1], nparray[1:]), 2.0)


def computeEntropy_binary(label_col):
    # assume the class range has cardinality 2
    if not len(classLabelsRange) == 2:
        sys.exit('ERROR: non-binary class labels.')

    # count the frequency for one class
    count = 0
    for label in label_col:
        if label == classLabelsRange[0]:
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


def computeEntropy_dividedSet(data_divded, data_whole):
    '''
    Compute the entropy for a list of data subsets
    '''
    if len(data_whole) == 0:
        return 0
    entropy = 0
    # for each subset
    for data_sub in data_divded:
        # collect all class labels
        classLabels = []
        for instance in data_sub:
            classLabels.append(instance[-1])

        # weight the entropy by the occurence
        p_x = 1.0 * len(data_sub) / len(data_whole)
        entropy += p_x * computeEntropy_binary(classLabels)
    return entropy


def findBestThreshold(data_, featureIdx_):
    # if len(data_) == 0: raise Exception('ERROR: the input data set is empty')
    feature_values = []
    for instance in data_:
        feature_values.append(instance[featureIdx_])

    allThresholds = neighbourMean(np.array(feature_values))
    # find the threshold (split) with the lowest entropy
    bestThreshold = 0
    minEntropy = float('inf')
    for t in range(len(allThresholds)):
        # split the data with the t-th threshold
        data_divded = splitData_continuous(data_, featureIdx_, allThresholds[t])
        entropy_temp = computeEntropy_dividedSet(data_divded, data_)
        # keep track of the min entropy and its index
        # when there is a tie, pick the first one, achieved by < (instead of <=)
        if entropy_temp < minEntropy:
            minEntropy = entropy_temp
            bestThreshold = allThresholds[t]
    return minEntropy, bestThreshold



def computeConditionalEntropy(data_, feature_used_):
    # loop over all features
    nFeatures = len(metadata.types())-1
    entropy_YgX = np.zeros(nFeatures)
    entropy_YgX.fill(np.NaN)

    for i in range(nFeatures):
        # skip features that is alread used
        if feature_used_[i]:
            continue
        # condition 1: numeric feature
        if isNumeric(metadata.types()[i]):
            entropy, bestThreshold_idx = findBestThreshold(data_, i)

        # condition 2: nominal feature
        else:
            # split the data with the ith feature
            data_divded = splitData_discrete(data_, i, feature_range[i])
            # accumulate entropy
            entropy = computeEntropy_dividedSet(data_divded, data_)

        # save the entropy
        entropy_YgX[i] = entropy
    return entropy_YgX


def computInfoGain(data_, feature_used_):
    # compute information gain for all features
    yLabels_ = getLabelsFromData(data_)
    entropy_Y = computeEntropy_binary(yLabels_)
    entropy_YgX = computeConditionalEntropy(data_, feature_used_)
    infomationGain = np.subtract(entropy_Y, entropy_YgX)
    return infomationGain


###################### TREE FUNCTIONS ##########################


def stoppingGrowing(data_, infomationGain_, feature_used_):
    def setIsPure(data_):
        labels = []
        # collect all class labels
        for instance in data_:
            labels.append(instance[-1])
        labels = set(labels)
        # check if set cardinality == 1
        if len(labels) == 1:
            return True
        return False

    # remove nan value
    infomationGain_ = np.array(infomationGain_)
    infomationGain_ = infomationGain_[~np.isnan(infomationGain_)]

    # (i) all of the training instances reaching the node belong to the same class
    # (ii) number of training instances reaching the node < m
    # (iii) no feature has positive information gain
    # (iv) there are no more remaining candidate splits at the node.
    if setIsPure(data_) or len(data_) < m or all(sp.less(infomationGain_, 0)) or all(feature_used_):
        return True
    return False


def findBestSplit(data_, feature_used_):
    '''
    Return feature that maximize information gain
    select the best feature (used feature will have NAN as its infoGain)
    '''
    infomationGain = computInfoGain(data_, feature_used_)
    best_feature_index = np.nanargmax(infomationGain)
    return best_feature_index


def getLabelCounts(data_):
    if not len(classLabelsRange) == 2:
        raise Exception('ERROR:non-binary class detected')
    count1 = 0
    count2 = 0
    # loop over all training example, count the occurence of each class value
    for instance in data_:
        if instance[-1] == classLabelsRange[0]:
            count1 += 1
        elif instance[-1] == classLabelsRange[1]:
            count2 += 1
        else:
            raise Exception('ERROR:un-recognizable class label for the training example')
    return count1, count2

def getMajorityClass(data_, parent_):
    '''
    get the majority vote for the current input data set
    if there is a tie, choose the first class listed in "classLabelsRange"
    the tie is handled by np.argmax()
    :param data_: the current data set
    :param classLabelsRange: all possible values for the class, should be binary
    :return: the majority vote
    '''
    count1, count2 = getLabelCounts(data_)
    # return the majority
    if count1 > count2:
        majorityClassLabel = classLabelsRange[0]
    elif count1 < count2:
        majorityClassLabel = classLabelsRange[1]
    else:
        majorityClassLabel = parent_.getClassification()

    return majorityClassLabel, [count1, count2]



def initTreeNode(data_, feature_name_, feature_used_, feature_val_, parent_, isRoot_, isLeaf,
                 isLeftChild_ = False):
    # read some attributes
    n_feature_val = feature_val_
    n_feature_used = feature_used_
    n_parent = parent_
    n_feature_name = feature_name_
    if isRoot_:
        count1, count2 = getLabelCounts(data_)
        if count1 >= count2:
            n_label = classLabelsRange[0]
        else:
            n_label = classLabelsRange[1]
        counts = [count1, count2]
    else:
        n_label, counts = getMajorityClass(data_, n_parent)

    # make the node
    node_temp = decisionTreeNode()
    node_temp.setFeature_belong(n_feature_name, n_feature_val)
    node_temp.updateUsedFeature(n_feature_used)
    node_temp.setParent(n_parent)

    # set classification data
    node_temp.setClassificationLabel(n_label)
    node_temp.setLabelCounts(counts)

    # set node type
    node_temp.setToRoot(isRoot_)
    if isLeaf:
        node_temp.setToLeaf()
    if isLeftChild_:
        node_temp.setToLeftChild()
    return node_temp


def recordUsedFeature_discrete(feature_used_, best_feature_idx_):
    # if the input feature is discrete, then we can nolonger split on it
    if not isNumeric(metadata.types()[best_feature_idx_]):
        feature_used_[best_feature_idx_] = True
    else:
        raise Exception('The feature is continuous')
    return feature_used_


def recordUsedFeature_continuous(data_,best_feature_idx_,feature_used_):
    list = []
    for instance in data_:
        list.append(instance[best_feature_idx_])
    list = set(list)
    if len(list) == 1:
        feature_used_[best_feature_idx_] = True
    return feature_used_


def makeSubtree(data_, feature_name_belong, feature_used_, feature_val_cur_,
                parent_ = None, isRoot_ = True, isLeftChild_ = False):

    # compute the current info gain for the stop check
    infomationGain = computInfoGain(data_, feature_used_)

    if stoppingGrowing(data_, infomationGain, feature_used_):
        # make a leaf node N
        isLeaf = True
        leaf = initTreeNode(data_, feature_name_belong, feature_used_, feature_val_cur_, parent_,
                            isRoot_, isLeaf, isLeftChild_)
        return leaf

    else:
        isLeaf = False
        # pick the best feature
        best_feature_idx = findBestSplit(data_, feature_used_)

        # start creating the tree
        node = initTreeNode(data_, feature_name_belong, feature_used_, feature_val_cur_, parent_,
                            isRoot_, isLeaf, isLeftChild_)
        feature_name_belong = metadata.names()[best_feature_idx]

        # split the data using the best feature
        # handle continous and discrete feature separately
        if isNumeric( metadata.types()[best_feature_idx]):
            _, bestThreshold = findBestThreshold(data_, best_feature_idx)
            [data_left, data_right] = splitData_continuous(data_, best_feature_idx, bestThreshold)

            # make the left child
            if not len(data_left) == 0: # TODO remove this test
                # create a child note
                child_left = makeSubtree(data_left, feature_name_belong, feature_used,
                                         bestThreshold, node, False, True)
                node.setChildren(child_left)

            # make the right child
            if not len(data_right) == 0:
                # create a child note
                child_right = makeSubtree(data_right, feature_name_belong, feature_used,
                                          bestThreshold, node, False)
                node.setChildren(child_right)

        # for discrete feature
        else:
            # split the data
            data_divided = splitData_discrete(data_, best_feature_idx, feature_range[best_feature_idx])

            # for eachOutcome k of S, create children
            numPossibleFeatureValues = len(feature_range[best_feature_idx])
            for i in range(numPossibleFeatureValues):
                # Dk = subset of instances that have outcome k
                feature_value_cur_i = feature_range[best_feature_idx][i]
                data_cur = data_divided[i]
                child_i = makeSubtree(data_cur, feature_name_belong, feature_used,
                                      feature_value_cur_i, node, False)
                node.setChildren(child_i)
    return node



def getLabelsFromData(data_):
    labels = []
    for instance in data_:
        labels.append(instance[-1])
    return labels



###################### visualization and test ##########################

def printNode(node, depth = 0):
    # print the infomation for the current node
    if not node.isRoot:
        [count1, count2] = node.getLabelCounts()
        if isNumeric(metadata[node.getFeatureName()][0]):
            if node.isLeftChild:
                symbol = "<="
            else:
                symbol = ">"

            featureValue =  "%.6f" % node.getFeatureValue()
        else:
            symbol = "="
            featureValue = node.getFeatureValue()

        if node.isTerminalNode():
            print depth * "|\t" + "%s %s %s [%d %d]: %s" \
                                  % (node.getFeatureName(), symbol, featureValue,
                                     count1, count2, node.getClassification())
        else:
            print depth * "|\t" + "%s %s %s [%d %d]" \
                                  % (node.getFeatureName(), symbol, featureValue,
                                     count1, count2)
        depth +=1

    # recursively print other nodes
    for child in node.getChildren():
        printNode(child, depth)


def getAllFeatureNames():
    all_featureNames = []
    for name in metadata.names():
        all_featureNames.append(name)
    return all_featureNames


def classify(instance, node):
    prediction = None
    if node.isTerminalNode():
        return node.classification

    for child in node.children:
        featureName = child.getFeatureName()
        featureType = metadata[featureName][0]
        featureIndex = all_featureNames.index(featureName)

        if isNumeric(featureType):
            if child.isLeftChild:
                # print "%s <= %s" % (featureName, child.getFeatureValue())
                if instance[featureIndex] <= child.getFeatureValue():
                    prediction = classify(instance, child)

            else:
                # print "%s > %s" % (featureName, child.getFeatureValue())
                if instance[featureIndex] > child.getFeatureValue():
                    prediction = classify(instance, child)
        else:
            # print " %s = %s" % (featureName, child.getFeatureValue())
            if instance[featureIndex] == child.getFeatureValue():
                prediction = classify(instance, child)
    return prediction


def printTestPerformance(data_test, decisionTree, printResults = False):
    if printResults:
        print "<Predictions for the Test Set Instances>"
    correctCount = 0
    numData = len(data_test)
    for i in range(numData):
        instance = data_test[i]
        y_pred = classify(instance, decisionTree)
        y_actual = instance[-1]
        if y_pred == y_actual:
            correctCount +=1
        if printResults:
            print "%d: Actual: %s Predicted: %s" % (i+1, y_actual, y_pred)
    if printResults:
        print "Number of correctly classified: %d Total number of test instances: %d" \
          % (correctCount, numData)
    classification_accuracy = 1.0 * correctCount / numData
    return classification_accuracy


###################### END OF DEFINITIONS OF HELPER FUNCTIONS ##########################
visualizeResults = 1
# read input arguments
fname_train, fname_test, m = processInputArgs()
# load data
# feature_range [tuple] = the possible values for discrete feature, None for continous
# feature_vals_unique [list] = the observed values for all features
# columns [list] = the actual feature vector from data_train
data_train, metadata, feature_range, feature_vals_unique = loadData(fname_train)

# read some parameters
feature_used = np.zeros((len(metadata.types()) - 1,), dtype=bool)
classLabelsRange = feature_range[-1]
all_featureNames = getAllFeatureNames()


# # build the tree
feature_val_cur = None
feature_name = None

decisionTree = makeSubtree(data_train, feature_name, feature_used, feature_val_cur)
# show the tree and print the performance
printNode(decisionTree)

# show test set performance
data_test, _, _, _= loadData(fname_test)
printTestPerformance(data_test, decisionTree, True)





################################ Problem 2 ################################
# get data
def randomSample_fromData(data, proportion):
    numData = len(data)
    num_training_data = np.round(proportion * numData)
    # take a random subet of the training data
    temp_idx = np.random.choice(numData, num_training_data, replace=False)
    subset_train_data = []
    for idx in temp_idx:
        subset_train_data.append(data[idx])
    # formatting
    subset_train_data = np.array(subset_train_data)
    return subset_train_data

# plot the performance
if visualizeResults:
    import matplotlib.pyplot as plt
    m = 4
    # Plot points for training set sizes that represent 5%, 10%, 20%, 50% and 100% of the instances in
    # each given training file. For each training-set size (except the largest one), randomly draw 10
    # different training sets and evaluate each resulting decision tree model on the test set. For each
    # training set size, plot the average test-set accuracy and the minimum and maximum test-set
    # accuracy. Be sure to label the axes of your plots. Set the stopping criterion m=4 for these
    # experiments.

    simSize = 10
    prop_training_data = np.array([.05, .1, .2, .5, 1])
    numConditions = len(prop_training_data)
    numData = len(data_train)
    num_training_data = np.round(prop_training_data * numData)

    # preallocate
    accuracy = np.zeros((simSize, numConditions))

    for j in range(simSize):
        for i in range(numConditions):
            proportion = prop_training_data[i]
            subset_data_train = randomSample_fromData(data_train, proportion)
            temp_dt = makeSubtree(subset_data_train, feature_name, feature_used, feature_val_cur)
            accuracy[j,i] = printTestPerformance(data_test, temp_dt)
            del temp_dt

    # accuracy = np.reshape(range(25), (5,5))
    accuracy_mean = np.mean(accuracy, 0)
    accuracy_max = np.amax(accuracy,0)
    accuracy_min = np.amin(accuracy,0)

    plt.figure(1)
    plt.plot(range(numConditions), accuracy_mean)
    plt.plot(range(numConditions), accuracy_max)
    plt.plot(range(numConditions), accuracy_min)

    plt.xticks(range(numConditions), prop_training_data)
    plt.title('%s' % fname_test)
    plt.ylabel('Test set classification accuracy')
    plt.xlabel('Proportion of training data')
    plt.show()



    ################################ Problem 3 ################################
    # For this part, you will investigate how predictive accuracy varies as a function of tree size.
    # Using the entire training set. Plot curves showing how test-set accuracy varies with the
    # value m used in the stopping criteria.
    # Show points for m = 2, 5, 10 and 20. Be sure to label the axes of your plots.
    parameters = np.array([2,5,10,20])
    accuracy = np.zeros((len(parameters),))
    for i in range(len(parameters)):
        m = parameters[i]
        temp_dt = makeSubtree(data_train, feature_name, feature_used, feature_val_cur)
        accuracy[i] = printTestPerformance(data_test, temp_dt)

    plt.figure(2)
    plt.plot(range(len(parameters)), accuracy)

    plt.xticks(range(len(parameters)), parameters)
    plt.title('%s' % fname_test)
    plt.ylabel('Test set classification accuracy')
    plt.xlabel('m')
    plt.show()