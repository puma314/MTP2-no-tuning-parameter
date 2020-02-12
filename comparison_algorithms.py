import os
import numpy as np
import uuid
import sklearn.linear_model

def get_uuid():
    return uuid.UUID(bytes=os.urandom(16), version=4)

def CLIME(X):
    X = X - np.mean(X,axis=0)
    uid = get_uuid()
    in_name = os.path.join(os.getcwd(), "rscripts", "clime_in_{}.npy".format(uid))
    out_name = os.path.join(os.getcwd(), "rscripts", "clime_out_{}.npy".format(uid))
    np.save(in_name, X)
    args = ['Rscript', 'rscripts/clime.R', str(uid)]
    p = Popen(args, stdout=PIPE)
    output = []
    while p.poll() is None:
        lin = p.stdout.readline()
        output.append(lin)
        print(lin)
    output_str = [x.decode('utf-8') for x in output] 
    if 'Error in solve.default(Sigma)' in ''.join(output_str):
        return None
    omega = np.load(out_name)
    os.remove(in_name)
    os.remove(out_name)
    hypothesis_graph = np.ones((p,p))
    for i in range(p):
        for j in range(p):
            if np.isclose(omega[i,j], 0, atol=1e-8):
                hypothesis_graph[i,j] = 0
    return hypothesis_graph


def TIGER(X):
    uid = get_uuid()
    X = X - np.mean(X, axis = 0)
    np.save(os.path.join(os.getcwd(), "rscripts", "tiger_in_{}.npy".format(uid)), X)
    args = ['Rscript', 'rscripts/tiger.R', str(uid)]
    p = Popen(args, stdout=PIPE)
    while p.poll() is None:
        print(p.stdout.readline())
    omega = np.load(os.path.join(os.getcwd(), "tiger_out_{}.npy".format(uid)))
    os.remove(os.path.join(os.getcwd(), "rscripts", "tiger_in_{}.npy".format(uid)))
    os.remove(os.path.join(os.getcwd(), "rscripts", "tiger_out_{}.npy".format(uid)))
    hypothesis_graph = np.ones((p,p))
    for i in range(p):
        for j in range(p):
            if np.isclose(omega[i,j], 0, atol=1e-8):
                hypothesis_graph[i,j] = 0
    return hypothesis_graph


def CMIT(X, xi, eta):
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
                    pc = np.abs(partial_cov(sample_cov, subset_i_j, i, j))
                    if pc <= xi:
                        edge_exists = False
                        break
            if edge_exists:
                hypothesis_graph[i,j] = 1
                hypothesis_graph[j,i] = 1
    
    return hypothesis_graph

def nbsel(X, lamb):
    N, p = X.shape
    model = sklearn.linear_model.Lasso(alpha = lamb)
    res = np.zeros((p,p))
    for i in range(p):
        node = X[:, i]
        assert node.shape == (N,)
        if i == 0:
            other_nodes = X[:, 1:]
        elif i == p-1:
            other_nodes = X[:, :p-1]
        else:
            other_nodes = np.hstack((X[:,:i], X[:,i+1:]))
        assert other_nodes.shape == (N, p-1)
        model.fit(other_nodes, node)
        other_nodes_list = list(range(p))
        other_nodes_list.remove(i)
        for on, coef in zip(other_nodes_list, model.coef_):
            if coef != 0.:
                res[i, on] = 1
                res[on, i] = 1
    return res

"""
def SH(data, lambdas):
    cov = np.cov(data.T)
    prec = run_single_MTP(cov)
    results = {}

    for q in lambdas:
        results[q] = attr_threshold_new(prec, q)

    return results, prec

def run_single_MTP(sample_cov):
    #mkdir MTP2-finance/matlab/data
    og_dir = os.getcwd()
    try:
        os.chdir("../MTP2-finance/matlab")
        mdict = {'S': sample_cov}
        inp_path = './data/algo_sample_cov.mat'
        out_path = './data/algo_est.mat'
        scipy.io.savemat(inp_path, mdict)
        command = "matlab -nodisplay -nodesktop -r \"computeomega '{}' '{}'; exit;\"".format(inp_path, out_path)
        os.system(command)
        ans = scipy.io.loadmat('./data/algo_est.mat')['Omega']
    finally: 
        os.chdir(og_dir)
    return ans

def attr_threshold_new(prec, q):
    new_prec = prec.copy()
    p, _ = prec.shape
    off_diags = []
    for i in range(p):
        for j in range(i+1, p):
            if prec[i, j] != 0:
                off_diags.append(prec[i,j])
    thres = np.quantile(off_diags, q, interpolation='nearest')
    for i in range(p):
        for j in range(i+1, p):
            ele = prec[i, j]
            if -ele > abs(thres):
                pass
            else:
                #print(i,j)
                new_prec[i, j] = 0
                new_prec[j, i] = 0
    return new_prec
"""

### Stability selection related

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

def stability_wrapper(algo, **algorithm_args):
    """Given an algorithm, returns a stability selection version of it."""
    def f(X, lambdas, pi, num_subsamples):
        results, probs = stability_selection(algo, X, lambdas, num_subsamples)
        res = get_stability_edges(probs, lambdas, pi)
        n, p = data.shape
        omega = np.zeros((p,p))
        for e in res:
            omega[e] = 1
            omega[e[::-1]] = 1
        return omega, results, probs
    return f

def stability_nbsel(X, **kwargs):
    return stability_wrapper(nbsel)(X, **kwargs)

def stability_CMIT(X, **kwargs):
    return stability_wrapper(CMIT)(X, **kwargs)

def stability_glasso(X, **kwargs):
    return stability_wrapper(glasso)(X, **kwargs)
