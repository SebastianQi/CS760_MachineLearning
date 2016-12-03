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
    CVB_sizes = np.array(CVB_sizes)
    CVB_nPos = np.array(CVB_nPos)
    return CVB_sizes, CVB_nPos, CVB_sizes-CVB_nPos


def getCVB_idx(M,K,CVB_nPos, CVB_nNeg):
    # assume binary classes
    cvidx_unused = np.array(range(M))
    cvidx_unused_neg = cvidx_unused[Y == 0]
    cvidx_unused_pos = cvidx_unused[Y == 1]

    # set up cv idx for all blocks
    cvidx_bool = np.zeros((M, K), dtype=bool)
    for k in range(K):
        # take random sample w/o rep
        temp_idx_pos = np.random.choice(cvidx_unused_pos, CVB_nPos[k], replace=False)
        temp_idx_neg = np.random.choice(cvidx_unused_neg, CVB_nNeg[k], replace=False)

        # remove the selected index
        cvidx_unused_pos = np.setdiff1d(cvidx_unused_pos, temp_idx_pos)
        cvidx_unused_neg = np.setdiff1d(cvidx_unused_neg, temp_idx_neg)
        #
        cvidx_bool[temp_idx_pos, k] = True
        cvidx_bool[temp_idx_neg, k] = True
    return cvidx_bool



### start
K = 10
fname = 'chess-KingRookVKingPawn.arff.txt'
X, Y, metadata, numVals = loadData(fname)
[M, N] = np.shape(X)

# do 10 folds cross validation, for NB and TAN
CVB_sizes, CVB_nPos, CVB_nNeg = getCVPartition(K, M, Y)
CVB_idx = getCVB_idx(M,K,CVB_nPos, CVB_nNeg)

print np.sum(CVB_idx,0)
print np.sum(CVB_idx,1)
print np.sum(CVB_idx)

# compare the performance - T test
accuracy = np.zeros((K,2))
for k in range(K):
    print k
    train_idx = np.invert(CVB_idx[:, k])
    test_idx = CVB_idx[:, k]
    X_train = X[train_idx,:]
    Y_train = Y[train_idx]
    X_test = X[test_idx,:]
    Y_test = Y[test_idx]

    # fit Naive Bayes
    P_Y, P_XgY = buildNaiveBayesNet(X_train, Y_train, numVals)
    Y_hat_nb, _ = computePredictions_NaiveBayes(X_test, P_Y, P_XgY)
    accuracy[k, 0] = 1.0 * sum(Y_hat_nb == Y_test) / len(Y_test)

    # fit TAN
    MST = computeTanStructure(X_train, Y_train, numVals)
    CPT = buildTreeAugBayesNet(X_train, Y_train, numVals, MST)
    Y_hat_tan, _ = computePredictions_TAN(X_test, CPT, MST, numVals)
    accuracy[k, 1] = 1.0 * sum(Y_hat_tan == Y_test) / len(Y_test)

print accuracy

##
print '---'
print M,N
print 'num total pos', np.sum(Y)
print '10 x CVB_pos', np.sum(CVB_nPos)

print CVB_sizes
print CVB_nPos
print CVB_nNeg
