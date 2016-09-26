import numpy as np
import scipy as sp


def neighbourMean(nparray):
    '''
    Find all possible threshold values given the a set of ordered feature values
    :param nparray: order continuous valued feature vector
    :return: thresholds
    '''
    return np.divide(np.add(nparray[0:-1], nparray[1:]), 2.0)


x = 111.500000


val =  "%.6f" % x
val = str(val)
print val


