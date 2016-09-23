from util import *

# tree node for ID3 classification tree
class decisionTreeNode:
    def __init__(self, feature = None , feature_type = None, feature_val = None,
                 parent = None, children = [], classification = None, feature_used = None):
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

    # getters
    def getChildren(self):
        return self.children

    def getClassification(self):
        return self.classification

    def getParent(self):
        return self.parent

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