from bayesNetAlg import *
from util import *

def getCVPartition(K, M, Y):
    # generate a 10 folds partition
    percentPos = 1.0*np.sum(Y)/len(Y)
    CVB_size = round(1.0*M / K)
    CVB_pos = round(CVB_size * percentPos)

    M_temp = M
    totalPos = np.sum(Y)
    CVB_sizes = []
    CVB_nPos = []
    for i in range(K):
        if M_temp > CVB_size:
            M_temp -= CVB_size
            totalPos -= CVB_pos
            CVB_sizes.append(CVB_size)
            CVB_nPos.append(CVB_pos)
        else:
            CVB_sizes.append(M_temp)
            CVB_nPos.append(totalPos)
    return CVB_sizes, CVB_nPos

### start
K = 10
fname = 'chess-KingRookVKingPawn.arff.txt'

# load data
X, Y, metadata, numVals = loadData(fname)
[M, N] = np.shape(X)

# do 10 folds cross validation, for NB and TAN
CVB_sizes, CVB_nPos = getCVPartition(K, M, Y)


# compare the performance - T test





print M,N
print 'num total pos', np.sum(Y)
print '10 x CVB_pos', np.sum(CVB_nPos)

print CVB_sizes
print CVB_nPos

