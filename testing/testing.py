import numpy as np
import scipy as sp
import sys


x = np.reshape(range(12), (3,4))
print x
print np.delete(x,0,0)
print x[0,:]
print x


#
# def getKLargestElements(array, K):
#     # ind = np.argpartition(array, -K)[-K:]
#     ind = np.argpartition(array, K)
#     print array
#     print array[ind[:K]]
#     sys.exit('STOP')
#     return ind, array[ind]
#
# K = 4
#
# a = np.array([1, 2, 3, 4, 5])
#
# for i in range(1):
#     # a = np.random.rand(20,)
#     ind, _ = getKLargestElements(a,K)
#     print type(ind)
#     sys.exit('STOP')
#
#     if not (ind == np.argmax(a)):
#         print 'error detected'
#
# print 'done'
#
