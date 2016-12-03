import numpy as np
# prim - minimum spanning tree
# Tim Wilson, 2-25-2002

#A = adjacency matrix, u = vertex u
def adjacent(A, u):
    L = []
    for x in range(len(A)):
        if A[u][x] > 0 and x <> u:
            L.insert(0,x)
    L = sorted(L) # sort by increasing order, s.t. earlier vertex will be preferred
    return L

#Q = min queue
def extractMin(Q):
    q = Q[0]
    Q.remove(Q[0])
    return q

#Q = min queue, V = vertex list
def sort_increaseKey(Q, K):
    for i in range(len(Q)):
        for j in range(len(Q)):
            if K[Q[i]] > K[Q[j]]:
                s = Q[i]
                Q[i] = Q[j]
                Q[j] = s

#V = vertex list, A = adjacency list, r = root
def findMaxSpanningTree_prim(A):
    V = range(np.shape(A)[0])
    # initialize and set each value of the array P (pi) to none
    # pi holds the parent of u, so P(v)=u means u is the parent of v
    P = [None]*len(V)
    # initialize and set each value of the array K (key) to inf
    K = [-float('inf')]*len(V)
    # initialize the min queue and fill it with all vertices in V
    Q = V
    # set the key of the root (the 1st node) to 0
    K[0] = 0
    sort_increaseKey(Q, K)    # maintain the min queue

    # loop while the min queue is not empty
    while len(Q) > 0:
        u = extractMin(Q)    # pop the first vertex off the min queue
        # loop through the vertices adjacent to u
        adjVertex = adjacent(A, u)
        for v in adjVertex:
            w = A[u][v]    # get the weight of the edge uv
            # proceed if v is in Q and the weight of uv is less than v's key
            if Q.count(v)> 0 and w > K[v]:
                P[v] = u    # set v's parent to u
                K[v] = w    # v's key to the weight of uv
                sort_increaseKey(Q, K)    # maintain the min queue
    return P