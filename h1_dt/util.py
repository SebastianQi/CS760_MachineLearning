# CONSTANT
TYPE_NUMERIC = "numeric"
TYPE_REAL = "real"
TYPE_INT = "integer"
ATTRIBUTE_INDICATOR = "@attribute"

# SMALL_NUMBER = 1e-10

######################## HELPER FUNCTIONS ########################
def isNumeric(type_str):
    if type_str == TYPE_NUMERIC or type_str == TYPE_REAL or type_str == TYPE_INT:
        return True
    else:
        return False

######################## CHECKER FUNCTIONS ########################

def printAllFeatures(metadata, feature_vals):
    for i in range(len(feature_vals)):
        if isNumeric(metadata.types()[i]):
            featurevalues = "[......]"
            # featurevalues = str(feature_vals[i])
        else:
            featurevalues = str(feature_vals[i])

        print "%d\t%8s\t%s\t%s" % \
              (i, metadata.names()[i], metadata.types()[i], featurevalues)
    print 70*"-"

######################## NOT USED FUNCTIONS, TESTERS ########################




def testSplitData_continous(featureIdx, threshold):
    # call the function
    data_divided = splitData_continuous(data_train, featureIdx, threshold)
    # pint the split
    for data_sub in data_divided:
        for instance in data_sub:
            print instance[featureIdx]
        print "\n"

def testSplitData_discrete(featureIdx):
    # call the function
    data_divided = splitData_discrete(data_train, featureIdx, feature_vals[featureIdx])
    # pint the split
    for data_sub in data_divided:
        for instance in data_sub:
            print instance[featureIdx]
        print "\n"


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
#             if isNumeric(range_str[0]):
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