class decisionTreeNode:
    def __init__(self, feature = None , feature_type = None, feature_val = None,
                 children = None, label = None):
        self.feature = feature
        self.feature_type = feature_type
        self.feature_val = feature_val
        self.children = children
        self.label = label
