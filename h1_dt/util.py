# CONSTANT
TYPE_NUMERIC = "numeric"
TYPE_REAL = "real"
TYPE_INT = "integer"
ATTRIBUTE_INDICATOR = "@attribute"

######################## HELPER FUNCTIONS ########################
def isContinuous(type_str):
    if type_str == TYPE_NUMERIC or type_str == TYPE_REAL or type_str == TYPE_INT:
        return True
    else:
        return False



######################## NOT USED FUNCTIONS ########################

# def getFeatureRange(fname_train):
#     feature_range = []
#     feature_idx = 0
#     for line in open(fname_train):
#         if line.strip().startswith(ATTRIBUTE_INDICATOR):
#             # read the line
#             range_str = line.rstrip().split(' ')
#             # pop out line header and feature name
#             range_str.pop(0);range_str.pop(0)
#             # check if the feature is continuous or discrete
#             if isContinuous(range_str[0]):
#                 feature_range.append(TYPE_NUMERIC)
#             else:
#                 # remove the two brakets
#                 range_str.pop(0)
#                 range_str[-1] = range_str[-1][:-1]
#                 # remove commas
#                 for i in range(len(range_str)):
#                     range_str[i] = range_str[i].strip(',')
#                 # append the feature values
#                 feature_range.append(range_str)
#         feature_idx +=1
#     return feature_range