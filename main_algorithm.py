import numpy as np
import itertools

def partial_cov(S, subset, k, l):
    assert type(subset) == list
    assert sorted(subset) == subset
    ix_grid = np.ix_(subset, subset)
    submatrix = S[ix_grid]
    inv = np.linalg.inv(submatrix)
    k_idx = subset.index(k)
    l_idx = subset.index(l)
    return -inv[k_idx, l_idx]

def get_batch(X, B):
    assert type(B) == int
    N, p = X.shape
    subset_idx = np.random.choice(N, B, replace=False)
    ret = X[subset_idx]
    assert ret.shape == (B, p), ret.shape
    return ret

def get_edges(graph):
    all_edges = []
    p, _ = graph.shape
    for i in range(p):
        for j in range(i+1, p):
            #i < j
            if graph[i, j] == 1:
                all_edges.append((i,j))
    return all_edges

def adj(graph, i):
    adj = []
    for j, num in enumerate(graph[i]):
        if num == 1 and j != i:
            adj.append(j)
    return adj

def remove_adj_i(p, adj_i, i, j):
    poss = list(range(p))
    for x in adj_i:
        poss.remove(x)
    poss.remove(i)
    poss.remove(j)
    return poss

def no_tuning_parameters(X, gamma=0.85):
    """Implementation of Algorithm 1 in paper.

    This algorithm is a tuning parameter free algorithm which given
    observations (X) from a multivariate Gaussian distribution with
    precision matrix theta that is MTP2, estimates the underlying
    0-pattern of theta.

    Args:
        X: Observations.
        gamma: gamma in Algorithm 1.

    Returns:
        Estimate of 0-pattern of omega according to Algorithm 1.
    """
    N, p = X.shape
    M = int(np.power(N, gamma))
    print("N={}, M={}".format(N, M))

    # l is the size of the subset to consider.
    l = -1
    # We start by assuming all edges exist.
    hypothesis_graph = np.ones((p,p))

    valid_edge_exists = True

    while valid_edge_exists:
        valid_edge_exists = False
        l = l+1
        print("l={}".format(l))
        for edge in get_edges(hypothesis_graph):
            i, j = edge
            adj_i = adj(hypothesis_graph, i)
            if len(adj_i) >= l+1:
                # Only consider vertices with adjacency sets sufficiently
                # large.
                adj_i.remove(j)
                valid_edge_exists = True
            else:
                continue  # To next edge.

            stop = False
            subsets = list(itertools.combinations(adj_i, l))
            for S in subsets:
                S = list(S)
                all_K = remove_adj_i(p, S, i, j)
                for k in all_K:
                    if not stop:
                        subset = sorted(S + [k] + [i] + [j])
                        data = get_batch(X, M)
                        sample_cov = np.cov(data.T)
                        rho = partial_cov(sample_cov, subset, i, j)
                        if rho < 0:
                            # Delete (i,j) from graph.
                            hypothesis_graph[i,j] = 0
                            hypothesis_graph[j, i] = 0
                            stop = True
    return hypothesis_graph