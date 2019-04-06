import numpy as np

def isserlis(mat):
    #returns the isserling matrix of a given matrix
    #mat is a cov matrix
    p, _ = mat.shape
    sz = p*(p+1) // 2
    ret = np.zeros((sz, sz))
    countx = 0
    county = 0
    for i in range(p):
        for j in range(i,p):
            county = 0
            for u in range(p):
                for v in range(u, p):
                    entry = mat[i][u] * mat[j][v] + mat[i][v] * mat[j][u]
                    ret[countx][county] = entry
                    county += 1
            countx += 1
    return ret

def submatrix(mat, subset):
    assert(sorted(subset) == subset)
    return mat[subset, :][:, subset]

def mat_to_vec(mat):
    #given an symmetrix p by p matrix, returns the (p+1)*p/2 vector associated with the matrix 
    p, _ = mat.shape
    ret = np.zeros((p+1)*p // 2)
    count = 0
    for i in range(p):
        for j in range(i,p):
            ret[count] = mat[i][j]
            count += 1
    return ret

def subset_idx(u,v, N):
    #Given a subset N, return index of (u,v) in a matrix (with subcolumns belonging to only N)
    assert(sorted(N) == N) #N is sorted
    if u > v:
        u, v = v, u
        #print("Had to reverse u, v since they weren't sorted")
    p = len(N)
    
    count = 0
    for i in range(p):
        for j in range(i,p):
            if N[i] == u and N[j] == v:
                return count
            count += 1
    assert False, 'Error in subset_idx: {}, {} not in subset {}'.format(u, v, N)
    return None

def cov_to_corr(pred_cov):
    D = np.diag(np.sqrt(np.diag(pred_cov)))
    DInv = np.linalg.inv(D)
    R = DInv.dot(pred_cov).dot(DInv)
    return R

def assert_symm(mat):
    assert(np.linalg.norm(mat-mat.T) == 0)

def assert_PSD(cov):
    try:
        np.linalg.cholesky(cov)
    except:
        assert False, "Matrix not PSD!"

def get_adj(inv):
    #Given a precision matrix, generates the corresponding adjacency matrix
    adj = (inv != 0).astype(int)
    return adj

def get_S(adj):
    #Returns maximal degree
    S = np.max(np.sum(adj, axis = 0)) - 1
    return S