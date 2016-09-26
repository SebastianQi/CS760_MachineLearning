import numpy as np
import scipy as sp


def neighbourMean(nparray):
    '''
    Find all possible threshold values given the a set of ordered feature values
    :param nparray: order continuous valued feature vector
    :return: thresholds
    '''
    return np.divide(np.add(nparray[0:-1], nparray[1:]), 2.0)


x = range(1,10,1)
print x
print neighbourMean(x)