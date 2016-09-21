import numpy as np


array = range(10)
print array

means = np.zeros(len(array)-1)
for i in range(len(array)-1):
    means[i] = (array[i] + array[i+1]) / 2.0

print means

print np.divide(np.add(array[0:-1], array[1:]), 2.0)


list = []
for i in range(10):
    list.append(i)

array_list = np.array(list)



# for i in range(10):
#     print i
