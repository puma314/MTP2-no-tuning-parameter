import os
import numpy as np
import uuid
import sklearn.linear_model
import sklearn.covariance
from subprocess import Popen, PIPE
import warnings
import itertools
import main_algorithm
from scipy import io as sio

def get_uuid():
    return uuid.UUID(bytes=os.urandom(16), version=4)

def CLIME(X):
    X = X - np.mean(X,axis=0)
    _, p = X.shape
    uid = get_uuid()
    in_name = os.path.join(os.getcwd(), "rscripts", "clime_in_{}.npy".format(uid))
    out_name = os.path.join(os.getcwd(), "rscripts", "clime_out_{}.npy".format(uid))
    np.save(in_name, X)
    args = ['Rscript', 'rscripts/clime.R', str(uid)]
    process = Popen(args, stdout=PIPE)
    output = []
    while process.poll() is None:
        lin = process.stdout.readline()
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
    X = X - np.mean(X, axis = 0)
    _, p = X.shape
    uid = get_uuid()
    in_name = os.path.join(os.getcwd(), "rscripts", "tiger_in_{}.npy".format(uid))
    out_name = os.path.join(os.getcwd(), "rscripts", "tiger_out_{}.npy".format(uid))
    np.save(in_name, X)
    args = ['Rscript', 'rscripts/tiger.R', str(uid)]
    process = Popen(args, stdout=PIPE)
    while process.poll() is None:
        print(process.stdout.readline())
    omega = np.load(out_name)
    os.remove(in_name)
    os.remove(out_name)
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
                    pc = np.abs(main_algorithm.partial_cov(sample_cov, subset_i_j, i, j))
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

def glasso(data, lamb):
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
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

def SH(X, q):
    cov = np.cov(X.T)
    prec = run_single_MTP(cov)
    return attr_threshold_new(prec, q)

def run_single_MTP(sample_cov):
    #mkdir MTP2-finance/matlab/data
    og_dir = os.getcwd()
    try:
        os.chdir("./matlab")
        mdict = {'S': sample_cov}
        inp_path = './data/algo_sample_cov.mat'
        out_path = './data/algo_est.mat'
        sio.savemat(inp_path, mdict)
        command = "matlab -nodisplay -nodesktop -r \"computeomega '{}' '{}'; exit;\"".format(inp_path, out_path)
        os.system(command)
        ans = sio.loadmat('./data/algo_est.mat')['Omega']
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