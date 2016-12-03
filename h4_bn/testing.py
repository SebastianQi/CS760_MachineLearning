import numpy as np
from prim import *
from util import *
# import numpy.random

# print range(10,15,1)
# print np.random.choice(range(10,15,1), 3, replace=False)

x = np.reshape(np.array(range(12)), (4,3))
print x
idx = np.array([1,2,3])

print x[idx, 2]