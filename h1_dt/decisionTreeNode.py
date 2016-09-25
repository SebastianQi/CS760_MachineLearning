from util import *
import numpy as np

# tree node for ID3 classification tree
class decisionTreeNode:
    def __init__(self, feature = None , feature_type = None, feature_val = None,
                 parent = None, children = [], classification = None, feature_used = None,
                 isLeaf = False):
        '''
        :param feature:         the name of this feature
        :param feature_type:    nominal or numeric
        :param feature_val:     the value of this feature (if continuous)

        :param parent:          the parent node as a node
        :param children:        the children as a list
        :param label:           classification
        '''
        self.feature = feature
        self.feature_type = feature_type
        self.feature_val = feature_val
        self.parent = parent
        self.children = children
        self.classification = classification
        self.isLeaf = isLeaf
        self.feature_used = feature_used

        self.depth = 0
        self.dataCounts = np.array(2,)

    # getters
    def getChildren(self):
        return self.children

    def getClassification(self):
        return self.classification

    def getParent(self):
        return self.parent

    def getFeatureName(self):
        return self.feature

    def getUsedFeature(self):
        return self.feature_used

    def getFeatureType(self):
        return self.feature_type

    def getFeatureValue(self):
        return self.feature_val


    # setters
    def setFeature(self, _feature, _feature_type, _feature_val, _feature_used):
        # check input
        self.feature = _feature
        self.feature_type = _feature_type
        # if parent nominal, feature val the val of that nominal variable
        # else, it should be None
        # TODO OR if numeric, feature val should be a number, else, it should be None
        self.feature_val = _feature_val
        self.feature_used = _feature_used


    def setChildren(self, _children):
        self.children.append(_children)

    def setParent(self, _parent):
        self.parent = _parent

    def setClassificationLabel(self, _classLabel):
        self.classification = _classLabel

    def setToLeaf(self):
        self.isLeaf = True

    def setDepth(self, depth_):
        self.depth = depth_


    def isTerminalNode(self):
        if self.isLeaf:
            return True
        return False

    ## other methods

    def printInfo(self):

        # print "%s - %s - %s" % (str(self.feature), str(self.feature_type), str(self.feature_val))
        print "%s" % (str(self.feature))
        # print "\t%s" % ()

