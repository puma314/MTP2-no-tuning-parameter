import numpy as np
import itertools
import math

def partial_corr(S, subset, k, l):
    assert type(subset) == list
    assert sorted(subset) == subset
    ix_grid = np.ix_(subset, subset)
    submatrix = S[ix_grid]
    k_idx = subset.index(k)
    l_idx = subset.index(l)
    inv = np.linalg.inv(submatrix)
    return -inv[k_idx, l_idx] / np.sqrt(inv[k_idx, k_idx] * inv[l_idx, l_idx])

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

def our_algo(X, early_stop = None, verbose = False, true_sigma = None):
    if true_sigma is not None:
        omega = np.linalg.inv(true_sigma)
    else:
        omega = None

    N, p = X.shape
    M = int(np.power(N, 0.9))
    if not early_stop:
        early_stop = p

    l = 2
    edge_deleted = True
    hypothesis_graph = np.ones((p,p))

    while edge_deleted and l < early_stop:
        l = l+1
        print("Working on l = {}".format(l))
        all_subsets = list(itertools.combinations(range(p), l))
        edge_deleted = False
        for edge in get_edges(hypothesis_graph):
            i, j = edge
            rhos= []
            trues = []
            for s in all_subsets:
                if i not in s or j not in s:
                    continue
                s = list(s)
                batch = get_batch(X, M)
                sample_cov = np.cov(batch.T)

                if true_sigma is not None:
                    true = partial_corr(true_sigma, s, i, j)
                else:
                    true = None

                rho = partial_corr(sample_cov, s, i, j)
                rhos.append(rho)

                trues.append(true)
                if rho < 0:
                    #print('Deleted {} {}'.format(i, j))
                    if omega is not None and omega[i, j] < 0:
                        print("Deleted an edge that exists!", i, j)
                    #     if verbose:
                    #         print("False negative")
                        #print(list(zip(trues, rhos)))
                    edge_deleted = True
                    hypothesis_graph[i, j] = 0
                    hypothesis_graph[j, i] = 0
                    break
            #assert len(rhos) > 0, (len(rhos), i, j)
            if omega is not None and omega[i, j] == 0 and min(rhos) > 0:
                print("In this iteration, did not delete an edge that should have", i, j)
                #print(list(zip(trues, rhos)))
            #     if verbose:
            #         print("False positive")
                #print(trues)
                #print(rhos)
    return hypothesis_graph

def new_algo(X, m=0.85):
    N, p = X.shape
    M = int(np.power(N, m))
    print("Running new algorithm")
    print("N={}, M={}".format(N, M))
    l = -1
    hypothesis_graph = np.ones((p,p))

    valid_edge_exists = True

    while valid_edge_exists:
        valid_edge_exists = False
        l = l+1
        print("Working on l = {}".format(l))
        for edge in get_edges(hypothesis_graph):
            i, j = edge
            adj_i = adj(hypothesis_graph, i)
            if len(adj_i) >= l+1:
                adj_i.remove(j)
                valid_edge_exists = True
            else:
                continue #to next iteration

            stop = False
            combos = list(itertools.combinations(adj_i, l))
            rhos = []
            for S in combos:
                S = list(S)
                all_K = remove_adj_i(p, S, i, j)
                for k in all_K:
                    if not stop:
                        subset = sorted(S + [k] + [i] + [j])
                        data = get_batch(X, M)
                        #d = np.random.randint(K)
                        #data = batches[d]
                        sample_cov = np.cov(data.T)
                        rho = partial_cov(sample_cov, subset, i, j)
                        rhos.append(rho)
                        if rho < 0:
                            hypothesis_graph[i,j] = 0
                            hypothesis_graph[j, i] = 0
                            stop = True
    return hypothesis_graph

