import numpy as np
import matplotlib.pyplot as plt

def entropy_bernoulli(p):
    ent = -p * np.log(p) - (1 - p) * np.log(1 - p)
    return ent


step = .01
p = np.arange(step,1,step)
ent = entropy_bernoulli(p)

plt.plot(p, ent)
plt.ylabel('entropy - uncertainty')
plt.xlabel('p')
plt.show()




