# CONSTANT
TYPE_NUMERIC = "numeric"
TYPE_REAL = "real"
ATTRIBUTE_INDICATOR = "@attribute"

######################## HELPER FUNCTIONS ########################
def isContinuous(type_str):
    if type_str == TYPE_NUMERIC or type_str == TYPE_REAL:
        return True
    else:
        return False
