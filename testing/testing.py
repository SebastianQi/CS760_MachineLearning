import numpy as np
import scipy as sp
length = 10
feature_used = np.zeros((length,), dtype=bool)
index_feature_used = [0,2,4]
feature_used[index_feature_used] = True

# for i in range(length):
#     if feature_used[i]: continue
#     print i

x = [1,2,np.NaN]
x = np.array(x)
x = x[~np.isnan(x)]
print x
# print ~np.isnan(x)


print all(sp.greater(x, 0))