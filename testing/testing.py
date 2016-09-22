import numpy as np

x = range(10)
index = np.zeros((10,), dtype=bool)

index[3] = True


print x
x = np.array(x)
print index
print x[index]

# for i in range(10):
#     print i
