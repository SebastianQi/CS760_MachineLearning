from bayesNetAlg import *
from util import *


K = 10
fname = 'chess-KingRookVKingPawn.arff.txt'

# load data
X, Y, metadata, numVals = loadData(fname)
[M,N] = np.shape(X)


# generate a 10 folds partition

# do 10 folds cross validation, for NB and TAN

# compare the performance - T test





print M,N
print 'base rate =', 1.0*np.sum(Y)/len(Y)
for instance in X:
    print instance
    sys.exit('STOP')

