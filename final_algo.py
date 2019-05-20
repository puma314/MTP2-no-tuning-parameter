import numpy as np
import itertools
import math
from collections import defaultdict
import sklearn.linear_model
import sklearn.covariance
import os
import time
import scipy
import warnings

def get_algo_lambdas(M, eta, N, p):
    algo_lambdas = {
        'our': M,
        'SH': get_SH_lambdas(),
        'glasso': [0.1, 0.5, 1.0, 2., 8.],#long_lambdas(N, p),
        'nbsel': [0.1, 0.5, 1.0, 2., 8.],
        'anand': [(l, eta) for l in super_short_lambdas(N, p)]
    }
    return algo_lambdas

def long_lambdas(n,p):
    c = np.sqrt(np.log(p)/n)
    ls = []
    for a in [0.01, 0.05, 0.1, 0.3, 0.5, 1, 2, 4, 8, 16]:
        ls.append(a*c)
    return ls

def short_lambdas(n,p):
    c = np.sqrt(np.log(p)/n)
    ls = []
    for a in [0.01, 0.05, 0.1, 0.5, 1, 2, 8]:
        ls.append(a*c)
    return ls

def super_short_lambdas(n,p):
    c = np.sqrt(np.log(p)/n)
    ls = []
    for a in [0.1, 1, 8]:
        ls.append(a*c)
    return ls

def get_SH_lambdas():
    return [0.7, 0.8, 0.9, 0.95, 0.99, 1]

def GET_ALGOS(NUM_SUBSAMPLES):
    anand_stab = stability_wrapper(anandkumar_algo, NUM_SUBSAMPLES)
    nbsel_stab = stability_wrapper(nbsel, NUM_SUBSAMPLES)
    glasso_stab = stability_wrapper(glasso_vanilla, NUM_SUBSAMPLES)
    SH_stab = SH_stability_wrapper(NUM_SUBSAMPLES)
    algos = {
        'our': new_algo,
        'SH': SH_stab,
        'glasso': glasso_stab,
        'nbsel': nbsel_stab,
        'anand': anand_stab,
    }
    return algos

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

# def our_algo(X, early_stop = None, verbose = False, true_sigma = None):
#     if true_sigma is not None:
#         omega = np.linalg.inv(true_sigma)
#     else:
#         omega = None

#     N, p = X.shape
#     M = int(np.power(N, 0.9))
#     if not early_stop:
#         early_stop = p

#     l = 2
#     edge_deleted = True
#     hypothesis_graph = np.ones((p,p))

#     while edge_deleted and l < early_stop:
#         l = l+1
#         print("Working on l = {}".format(l))
#         all_subsets = list(itertools.combinations(range(p), l))
#         edge_deleted = False
#         for edge in get_edges(hypothesis_graph):
#             i, j = edge
#             rhos= []
#             trues = []
#             for s in all_subsets:
#                 if i not in s or j not in s:
#                     continue
#                 s = list(s)
#                 batch = get_batch(X, M)
#                 sample_cov = np.cov(batch.T)

#                 if true_sigma is not None:
#                     true = partial_corr(true_sigma, s, i, j)
#                 else:
#                     true = None

#                 rho = partial_corr(sample_cov, s, i, j)
#                 rhos.append(rho)

#                 trues.append(true)
#                 if rho < 0:
#                     #print('Deleted {} {}'.format(i, j))
#                     if omega is not None and omega[i, j] < 0:
#                         print("Deleted an edge that exists!", i, j)
#                     #     if verbose:
#                     #         print("False negative")
#                         #print(list(zip(trues, rhos)))
#                     edge_deleted = True
#                     hypothesis_graph[i, j] = 0
#                     hypothesis_graph[j, i] = 0
#                     break
#             #assert len(rhos) > 0, (len(rhos), i, j)
#             if omega is not None and omega[i, j] == 0 and min(rhos) > 0:
#                 print("In this iteration, did not delete an edge that should have", i, j)
#                 #print(list(zip(trues, rhos)))
#             #     if verbose:
#             #         print("False positive")
#                 #print(trues)
#                 #print(rhos)
#     return hypothesis_graph

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

def GET_ALGOS_ROC():
    return {
        'glasso': glasso_vanilla,
        'nbsel': nbsel,
        'SH': SH_lambda_wrapper,
        'our': new_algo,
        'anand': anandkumar_algo_lambda_wrapper
    }

def glasso_vanilla(data, lamb, KT=False):
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            if KT:
                cov = kendall_cov(data)
            else:
                cov = np.cov(data.T)
            glasso = sklearn.covariance.graphical_lasso(cov, alpha=lamb, mode='lars')
            if len(w) > 0 and issubclass(w[-1].category, sklearn.exceptions.ConvergenceWarning):
                #print(str(w[-1].message))
                print("graphical_lasso ConvergenceWarning: {}".format(lamb))
                return None
            _, omega_hat = glasso
            non_zero = np.nonzero(omega_hat)
            N, p = data.shape
            A = np.zeros((p,p))
            A[non_zero] = 1
            return A
            #return omega_hat
    except FloatingPointError:
        print("graphical_lasso FloatingPointError: {}".format(lamb))
        return None
    except OverflowError:
        print("graphical_lasso OverflowError: {}".format(lamb))
        return None

def graphical_lasso_CV(data, lambdas):
    N, p = data.shape
    train_data = data[:N//2]
    validation = data[N//2:]
    train_cov = np.cov(train_data.T)
    valid_cov = np.cov(validation.T)
    lls = []
    ests = {}
    for lamb in lambdas:
        glasso = graphical_lasso(train_cov, alpha=lamb, mode='lars')
        omega_hat = glasso.precision
        ests[lamb] = omega_hat
        _, logdet = np.linalg.slogdet(np.linalg.inv(omega_hat))
        ll = -N/2 * logdet - N/2*np.trace(np.matmul(valid_cov, omega-hat))
        lls.append((lamb, ll))

    best_lamb = list(sorted(lls, reverse=True, key = lambda x: x[1]))[0][0]

    return ests[best_lamb]

def SH_stability_wrapper(NUM_SUBSAMPLES):
    def SH_stability(data, lambdas, pi):
        print("IN SH STABILITY")
        N, p = data.shape
        MTP2_precs = []
        subN = N//2
        for _ in range(NUM_SUBSAMPLES):
            batch = get_batch(data, subN)
            import time
            start = time.time()
            print("running single MTP")
            MTP2res = run_single_MTP(np.cov(batch.T))
            end = time.time()
            print('done with single MTP', end-start)
            MTP2_precs.append(MTP2res)
        
        start = time.time()
        edges = []
        for i in range(p):
            for j in range(i+1, p):
                edges.append((i,j))
        
        results = defaultdict(list)
        probs = {}
        for thres in lambdas:
            for prec in MTP2_precs:
                results[thres].append(attr_threshold(prec, thres))

            probs[thres] = defaultdict(int)
            for res in results[thres]:
                for e in edges:
                    e_val = res[e]
                    if e_val < 0:
                        probs[thres][e] += 1
            for e in edges:
                probs[thres][e] /= len(results[thres])

        stable_edges = get_stability_edges(probs, lambdas, pi)
        omega = np.zeros((p,p))
        for e in stable_edges:
            omega[e] = 1
            omega[e[::-1]] = 1
        print(time.time() - start, 'HIIIIIIIII')
        return omega, results, probs, MTP2_precs
    return SH_stability


def stability_wrapper(algo, NUM_SUBSAMPLES=10):
    def f(data, lambdas, pi):
        results, probs = stability_selection(algo, data, lambdas, NUM_SUBSAMPLES)
        #print(probs)
        res = get_stability_edges(probs, lambdas, pi)
        # print(res)
        # print(len(res))
        # return res
        n, p = data.shape
        omega = np.zeros((p,p))
        for e in res:
            omega[e] = 1
            omega[e[::-1]] = 1
        return omega, results, probs
    return f

# def stability_anandkumar(data, lambdas, pi):
#     results, probs = stability_selection(anandkumar_algo, data, lambdas)
#     res = get_stability_edges(probs, lambdas, pi)
#     n, p = data.shape
#     omega = np.zeros((p,p))
#     for e in res:
#         omega[e] = 1
#         omega[e[::-1]] = 1
#     return omega

def anandkumar_algo_lambda_wrapper(X, lambdas):
    N, p = X.shape
    sample_cov = np.cov(X.T)
    assert sample_cov.shape == (p,p)

    partial_covs = {}
    results = {}
    for eta, xi in lambdas:
        print("Working on", eta, xi)
        hypothesis_graph = np.zeros((p,p))
        for i in range(p):
            for j in range(i+1, p):
                edge_exists = True
                #testing if edge (i,j) exists
                vertices = list(range(p))
                vertices.remove(i)
                vertices.remove(j)
                for l in range(1, eta+1):
                    if not edge_exists:
                        break
                    all_subsets = list(itertools.combinations(vertices, l))
                    for subset in all_subsets:
                        subset_i_j = sorted(subset + (i,j))
                        if tuple(subset_i_j) in partial_covs:
                            pc = partial_covs[tuple(subset_i_j)]
                        else:
                            pc = np.abs(partial_cov(sample_cov, subset_i_j, i, j))
                            partial_covs[tuple(subset_i_j)] = pc
                        if  pc <= xi:
                            edge_exists = False
                            break
                if edge_exists:
                    hypothesis_graph[i,j] = 1
                    hypothesis_graph[j,i] = 1
        results[(eta,xi)] = hypothesis_graph
    return results, (sample_cov, partial_covs)
    

def anandkumar_algo(X, xi, eta=2):
    print('Running anand with eta = {}'.format(eta))
    N, p = X.shape
    sample_cov = np.cov(X.T)
    assert sample_cov.shape == (p,p)

    hypothesis_graph = np.zeros((p,p))
    for i in range(p):
        for j in range(i+1, p):
            edge_exists = True
            #testing if edge (i,j) exists
            vertices = list(range(p))
            vertices.remove(i)
            vertices.remove(j)
            for l in range(1, eta+1):
                if not edge_exists:
                    break
                all_subsets = list(itertools.combinations(vertices, l))
                for subset in all_subsets:
                    subset_i_j = sorted(subset + (i,j))
                    if np.abs(partial_cov(sample_cov, subset_i_j, i, j)) <= xi:
                        edge_exists = False
                        break
            if edge_exists:
                hypothesis_graph[i,j] = 1
                hypothesis_graph[j,i] = 1
    
    return hypothesis_graph

def nbsel(data, lamb):
    N, p = data.shape
    model = sklearn.linear_model.Lasso(alpha = lamb)
    res = np.zeros((p,p))
    for i in range(p):
        node = data[:, i]
        assert node.shape == (N,)
        if i == 0:
            other_nodes = data[:, 1:]
        elif i == p-1:
            other_nodes = data[:, :p-1]
        else:
            other_nodes = np.hstack((data[:,:i], data[:,i+1:]))
        assert other_nodes.shape == (N, p-1)
        model.fit(other_nodes, node)
        other_nodes_list = list(range(p))
        other_nodes_list.remove(i)
        for on, coef in zip(other_nodes_list, model.coef_):
            if coef != 0.:
                res[i, on] = 1
                res[on, i] = 1
    return res

def stability_selection(algo, data, regularization_params, NUM_SUBSAMPLES=10):
    N, p = data.shape
    edges = []
    for i in range(p):
        for j in range(i+1, p):
            edges.append((i,j))
    subN = N//2
    results = {} #regulariation_param, [res_1, res_2, ... res_SUBSAMPLES]
    probs = {} #(lamb)(e) = prob of existing given lambda
    for lamb in regularization_params:
        print('Working on', lamb)
        results[lamb] = []
        for _ in range(NUM_SUBSAMPLES):
            batch = get_batch(data, subN)
            if hasattr(lamb, '__iter__'):
                res = algo(batch, *lamb)
            else:
                res = algo(batch, lamb)
            results[lamb].append(res)
            if res is None:
                break
        if results[lamb][-1] is not None:
            probs[lamb] = defaultdict(int)
            for res in results[lamb]:
                if res is None:
                    continue
                for e in edges:
                    e_val = res[e]
                    if e_val != 0.:
                        probs[lamb][e] += 1
            for e in edges:
                probs[lamb][e] /= len(results[lamb])
    return results, probs

def get_stability_edges(probs, lambs, pi):
    first_key = list(probs.keys())[0]
    edges = list(probs[first_key].keys())
    edges_plot = defaultdict(list)
    for e in edges:
        for l in lambs:
            if l in probs:
                edges_plot[e].append(probs[l][e])
    true_edges = set()
    for e, pis in edges_plot.items():
        if max(pis) >= pi:
            true_edges.add(e)
    return true_edges

def SH_lambda_wrapper(data, lambdas):
    cov = np.cov(data.T)
    prec = run_single_MTP(cov)
    results = {}

    for q in lambdas:
        results[q] = attr_threshold(prec, q)

    return results, prec

def run_single_MTP(sample_cov):
    og_dir = os.getcwd()
    try:
        os.chdir("../MTP2-finance/matlab")
        mdict = {'S': sample_cov}
        inp_path = './data/algo_sample_cov.mat'
        out_path = './data/algo_est.mat'
        scipy.io.savemat(inp_path, mdict)
        command = "matlab -nodisplay -nodesktop -r \"computeomega '{}' '{}'; exit;\"".format(inp_path, out_path)
        #print(command)
        os.system(command)
        ans = scipy.io.loadmat('./data/algo_est.mat')['Omega']
    finally: 
        os.chdir(og_dir)
    return ans

def attr_threshold(prec, q):
    new_prec = prec.copy()
    p, _ = prec.shape
    off_diags = []
    for i in range(p):
        for j in range(i+1, p):
            if prec[i, j] != 0:
                off_diags.append(prec[i,j])
    idx = int((1-q) * len(off_diags))
    sort = sorted(np.abs(off_diags))
    thres = sort[idx]
    #print(sort)
    #print(idx)
    #print(thres)
    for i in range(p):
        for j in range(i+1, p):
            ele = prec[i, j]
            if ele >= -thres:
                new_prec[i, j] = 0
                new_prec[j, i] = 0
    return new_prec