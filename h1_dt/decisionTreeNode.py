from util import *
import numpy as np

# tree node for ID3 classification tree
class decisionTreeNode:
    def __init__(self, feature_name_belong = None , feature_type_belong = None, feature_val_belong =
    None, feature_name_split = None, feature_type_split = None, parent = None,
                 classification = None, feature_used = None, isLeaf = False):
        '''
        :param feature:         the name of this feature
        :param feature_type:    nominal or numeric
        :param feature_val:     the value of this feature (if continuous)

        :param parent:          the parent node as a node
        :param children:        the children as a list
        :param label:           classification
        '''
        self.parent = parent
        self.children = []

        self.feature_name_belong = feature_name_belong
        self.feature_type_belong = feature_type_belong
        self.feature_val_belong = feature_val_belong
        self.feature_used = feature_used

        self.feature_name_split = feature_name_split
        self.feature_type_split = feature_type_split

        self.isLeaf = isLeaf
        self.isRoot = False
        self.isLeftChild = False

        self.depth = 0

        self.classification = classification
        self.labelCounts = np.array(2,)


    # getters
    def getChildren(self):
        return self.children

    def getClassification(self):
        return self.classification

    def getParent(self):
        return self.parent

    def getUsedFeature(self):
        return self.feature_used




    def getFeatureName(self):
        return self.feature_name_belong

    def getFeatureType(self):
        return self.feature_type_belong

    def getFeatureValue(self):
        return self.feature_val_belong

    def getFeatureName_split(self):
        return self.feature_name_split

    def isTerminalNode(self):
        if self.isLeaf:
            return True
        return False

    def isRoot(self):
        if self.isRoot:
            return True
        return False


    def getLabelCounts(self):
        return self.labelCounts


    def isLeftChild(self):
        return self.isLeftChild


    # setters

    def setToRoot(self, isRoot_):
        self.isRoot = isRoot_

    def setToLeftChild(self):
        self.isLeftChild = True

    def setFeature_belong(self, _feature_name, _feature_val):
        # check input
        # the feauture splitted by the parent node
        self.feature_name_belong = _feature_name
        # self.feature_type_belong = _feature_type
        self.feature_val_belong = _feature_val

        # if parent nominal, feature val the val of that nominal variable
        # else, it should be None
        # TODO OR if numeric, feature val should be a number, else, it should be None

    def setFeature_name_split(self, feature_name_split_):
        self.feature_name_split = feature_name_split_

    def setFeature_split(self, feature_name_split_, feature_type_split_):
        self.feature_name_split =feature_name_split_
        self.feature_type_split = feature_type_split_

    def updateUsedFeature(self, feature_used_):
        self.feature_used = feature_used_


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


    def setLabelCounts(self, counts):
        self.labelCounts = counts

