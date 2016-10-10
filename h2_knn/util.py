TYPE_NOMINAL = 'nominal'
TYPE_NUMERIC = 'numeric'
NAME_RESPONSE = 'response'

def printAllFeatures(metadata, feature_vals):
    for i in range(len(feature_vals)):
        featurevalues = "[......]"
        # featurevalues = str(feature_vals[i])
    print "%d\t%8s\t%s\t%s" % \
          (i, metadata.names()[i], metadata.types()[i], featurevalues)
    print 70*"-"
