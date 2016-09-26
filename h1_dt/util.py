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
