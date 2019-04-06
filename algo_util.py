import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import Util
from sklearn import linear_model
import random
def construct_D(p, N):
    assert(sorted(N) == N) #N is sorted
    l = len(N)
    sz1 = l * (l+1) // 2
    sz2 = p * (p+1) // 2
    D = np.zeros((sz1, sz2))
    D_row = 0
    count = 0
    for i in range(p):
        for j in range(i,p):
            if i in N and j in N:
                D[D_row, count] = 1
                D_row += 1
            count += 1
    return D

def generate_mat_sparse(p, target_S, sparse_thres = 0.7):
    S = target_S - 1
    count = 0
    while S != target_S:
        #print(count)
        count += 1
        if S > target_S:
            sparse_thres *= 1.15
        else:
            sparse_thres *= 0.85
        sam_inv = np.zeros((p,p))
        for i in range(p):
            for j in range(i+1, p):
                if np.random.uniform() < sparse_thres:
                    sam_inv[i][j] = 0
                else:
                    sam_inv[i][j] = np.random.uniform(low = -1., high = -0.2) 

        inv = np.minimum(sam_inv, sam_inv.T) #makes it symmetric
        adj = get_adj(inv + np.eye(p)) #for factoring in the diagonal to adjacency matrix calculation
        S = get_S(adj)

    for i in range(p): #sets it to be diagonally dominant
        s = abs(np.sum(inv[i]))
        inv[i][i] = s + np.random.uniform(low = 0.2, high = 1.)
    
    adj = get_adj(inv)
    S = get_S(adj)
    Util.assert_symm(inv)
    #print(S)
    assert(S == target_S)
    return inv

def generate_connected_mat(p, output = True):
    #generates a connected adjacency matrix for graph with p nodes
    is_connected = False
    tries = 0
    while not is_connected:
        tries += 1
        graph = nx.erdos_renyi_graph(p, 0.2)
        is_connected = nx.is_connected(graph)
    if output:
        print('Took {} tries'.format(tries))
        fig, ax = plt.subplots(1, 1, figsize=(8, 6));
        nx.draw(graph, with_labels = True)
    return nx.adjacency_matrix(graph).todense()

def adj_to_cov(adj):
    #Given adjacency matrix, constructs MTP2 precision matrix and returns inverse (covariance)
    prec = np.zeros(adj.shape)
    p = adj.shape[0]
    for i in range(p):
        for j in range(p):
            #print(i, j)
            #print(adj[i, j])
            if adj[i, j] != 0:
                #print(i, j)
                if j < i:
                    prec[i, j] = prec[j, i]
                elif j == i:
                    pass #set later
                else:
                    prec[i][j] = np.random.random() * -1
    for i in range(p):
        s = abs(np.sum(prec[i]))
        prec[i][i] = s + np.random.uniform(low = 0.2, high = 1)

    Util.assert_symm(prec)
    return np.linalg.inv(prec)

def find_random_vertices(adj, k, l, num = 1000):
    #adjacency matrix of graph
    #k, l are the vertices we want to regress on 
    #we are regressing l on k
    #num of random vertices we want
    #must be path between r and l that is not blocked by k or neigh(k)
    graph = nx.from_numpy_matrix(adj)
    neighs = graph[k]
    neighs_list = [n for n in neighs]
    for node in neighs_list:
        graph.remove_node(node)
    graph.remove_node(k)

    connected_comp = list(nx.node_connected_component(graph, l))
    connected_comp.pop(connected_comp.index(l))
    random.shuffle(connected_comp)
    return connected_comp[:min(num, len(connected_comp))]

def assemble_subsets(adj, k, l, num):
    r_list = find_random_vertices(adj, k, l, num)
    graph = nx.from_numpy_matrix(adj)
    neighs = [n for n in graph[k]]
    subsets = []
    for r in r_list:
        N = [k] + [l] + neighs + [r]
        subsets.append(sorted(N))
    return subsets

def sleppian(inp):
    cov = inp.copy()
    og_diag = np.diag(cov).copy()
    np.fill_diagonal(cov, 0)
    m = cov.max()
    fin = np.ones(cov.shape) * m
    np.fill_diagonal(fin, og_diag)
    return fin

def do_regression(sample, subset, target_vertex):
    LR = linear_model.LinearRegression(fit_intercept = False)
    data_subset = sample #use whole thing
    LR.fit(data_subset[:, subset], data_subset[:, target_vertex])
    
    to_remove = set()
    for i, c in enumerate(LR.coef_):
        if c <= 0:
            to_remove.add(subset[i])
    return to_remove

def get_partial_correlation(sample, subset, k, l):
    #get partial correlation betweek k, l given subset
    cov = np.cov(sample.T)
    _, p = sample.shape
    assert(cov.shape == (p, p))
    subset = subset + [k, l]
    subset = list(set(subset)) #as to not add k, l twice
    subset = sorted(subset)
    cov_subset = Util.submatrix(cov, subset)
    k_idx = subset.index(k)
    l_idx = subset.index(l)
    return np.linalg.inv(cov_subset)[k_idx, l_idx]

def get_connections(target_vertex, adj):
    true_connections = []
    not_connected = []
    for i, connect in enumerate(adj[target_vertex].tolist()[0]):
        if connect and i != target_vertex:
            true_connections.append(i)
        if not connect and i != target_vertex:
            not_connected.append(i)
    return true_connections, not_connected




