import numpy as np
import scipy.io
import os
import networkx as nx

def modularity(graph, sector):
    A = nx.adjacency_matrix(graph).todense()
    E = np.sum(A) // 2
    p = len(A)
    Q = 0
    for i in range(p):
        for j in range(p):
            if sector[i] == sector[j]:
                k_i = np.sum(A[i])
                k_j = np.sum(A[j])
                Q += A[i,j] - k_i*k_j / (2*E)
    return Q / (2*E)        

def confusion(hypothesis_graph, omega):
	TP = 0
	TN = 0
	FP = 0
	FN = 0
	
	p, _ = omega.shape
	for i in range(p):
		for j in range(i+1, p):
			#true positive
			o = omega[i,j]
			h = hypothesis_graph[i, j]
			h_is_0 = np.isclose(h, 0, atol = 1e-6)
			if o < 0 and not h_is_0:
				TP += 1
			elif o == 0 and h_is_0:
				TN += 1
			elif o < 0 and h_is_0:
				FN += 1
			elif o == 0 and not h_is_0:
				FP += 1
			else:
				assert False, "case bad"
	return TP, TN, FP, FN

def MCC(hypothesis_graph, omega):
	TP, TN, FP, FN = confusion(hypothesis_graph, omega)
	num = TP * TN - FP * FN
	denom = (TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)
	if denom == 0:
		return 0
	#print(TP, TN, FP, FN)
	#print(denom)
	return num / np.sqrt(denom)

######TO GENERATE GRAPHS#####


def grid_graph(p):
	def grid_adj(p):
		n = p * p
		M = np.zeros((n,n))
		for r in range(p):
			for c in range(p):
				i = r*p + c
				# Two inner diagonals
				if c > 0: M[i-1,i] = M[i,i-1] = 1
				# Two outer diagonals
				if r > 0: M[i-p,i] = M[i,i-p] = 1
		return M
	B = grid_adj(p)
	delta = np.real(sorted(np.linalg.eigvals(B))[-1])
	omega = 1.05 * delta * np.eye(B.shape[0]) - B
	sigma = np.linalg.inv(omega)
	D = np.diag(np.power(np.diag(sigma), -0.5))
	D_inv = np.linalg.inv(D)
	return D_inv.dot(omega).dot(D_inv)

def star(d, p):
	rho = 0.6 / np.power(d, 0.25) * np.hstack((np.ones(d), np.zeros(p-d-1)))
	D = 1 + np.sum(np.square(rho))
	row_0 = np.hstack(((D), -rho))
	rho_2d = np.expand_dims(rho, 1)
	rest = np.hstack((-rho_2d, np.eye(p-1)))
	return np.vstack((row_0, rest))

def their_random_graph(p, d):
	B = np.zeros((p,p))
	for i in range(p):
		for j in range(i+1, p):
			if np.random.rand() <= d/2:
				B[i,j] = 1
				B[j, i] = 1
	delta = np.real(sorted(np.linalg.eigvals(B))[-1])
	omega = 1.05 * delta * np.eye(B.shape[0]) - B
	sigma = np.linalg.inv(omega)
	D = np.diag(np.power(np.diag(sigma), -0.5))
	D_inv = np.linalg.inv(D)
	return D_inv.dot(omega).dot(D_inv)

def their_random_graph_2(p, d):
	B = np.zeros((p,p))
	for i in range(p):
		for j in range(i+1, p):
			if np.random.rand() <= d/2 and sum(np.abs(B[i, i+1:])) == 0:
				B[i,j] = np.random.rand() + 0.2
				B[j, i] = B[i,j]
	delta = np.real(sorted(np.linalg.eigvals(B))[-1])
	omega = 1.5 * delta * np.eye(B.shape[0]) - B
	return omega
	sigma = np.linalg.inv(omega)
	#D = np.diag(np.power(np.diag(sigma), -0.5))
	#D_inv = np.linalg.inv(D)
	#return D_inv.dot(omega).dot(D_inv)

def our_random_graph(p, d):
	omega = np.zeros((p,p))
	for i in range(p):
		for j in range(p):
			if i < j and np.random.rand() <= d:
				omega[i, j] = -1 * (np.random.rand() + 0.5)
				omega[j, i] = omega[i, j]
			if i == j:
				omega[i, j] = 1
	return omega

#########################################
###FOR RUNNING OTHER METHODS WERE COMPARING TO

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