# tree node for ID3 classification tree
class decisionTreeNode:
    def __init__(self, feature = None , feature_type = None, feature_val = None,
                 parent = None, children = [], classification = None):
        '''
        :param feature:         the name of this feature
        :param feature_type:    nominal or numeric
        :param feature_val:     the value of this feature
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
    def setFeature(self, _feature, _feature_type, _feature_val):
        self.feature = _feature
        self.feature_type = _feature_type
        self.feature_val = _feature_val

    def setChildren(self, _children):
        self.children(_children)

    def setParent(self, _parent):
        self.parent = _parent

    def setClassificationLabel(self, _classLabel):
        self.classification = _classLabel