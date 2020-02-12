import numpy as np
from utils_graphing import is_MTP2

def omega_modularity(A, sector):
	p = A.shape[0]
	for i in range(len(A)):
		A[i,i] = 0
	E = np.sum(A) // 2
	if E == 0:
		return 0
	#print('E', E)
	p = len(A)
	Q = 0
	for i in range(p):
		for j in range(p):
			if sector[i] == sector[j]:
				k_i = np.sum(A[i])
				k_j = np.sum(A[j])
				# print(i,j)
				# print(A[i,j])
				# print(k_i*k_j / (2*E))
				Q += A[i,j] - k_i*k_j / (2*E)
	return Q / (2*E) 

def modularity(graph, sector):
	A = nx.adjacency_matrix(graph).todense()
	for i in range(len(A)):
		A[i,i] = 0
	E = np.sum(A) // 2
	if E == 0:
		return 0
	#print('E', E)
	p = len(A)
	Q = 0
	for i in range(p):
		for j in range(p):
			if sector[i] == sector[j]:
				k_i = np.sum(A[i])
				k_j = np.sum(A[j])
				# print(i,j)
				# print(A[i,j])
				# print(k_i*k_j / (2*E))
				Q += A[i,j] - k_i*k_j / (2*E)
	return Q / (2*E)        

def confusion(hypothesis_graph, omega):
	assert is_MTP2(omega)
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
			o_is_0 = np.isclose(o, 0, atol = 1e-6)

			if not o_is_0 and not h_is_0:
				#it is present and hypothesis is present
				TP += 1
			elif o_is_0 and h_is_0:
				#it is not present and hypothesis is not present
				TN += 1
			elif not o_is_0 and h_is_0:
				#edge exists (positive), but we guess 0
				FN += 1
			elif o_is_0 and not h_is_0:
				#edge does not exist (negative) but we guess positive
				FP += 1
			else:
				print(hypothesis_graph, omega)
				assert False, "case bad"
	return TP, TN, FP, FN

def MCC_from_4(TP, TN, FP, FN):
	num = TP * TN - FP * FN
	denom = (TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)
	if denom == 0:
		return 0
	return num / np.sqrt(denom)

def MCC(hypothesis_graph, omega):
	if hypothesis_graph is None:
		print("MCC returned 0 because hypothesis_graph is None")
		return 0
	TP, TN, FP, FN = confusion(hypothesis_graph, omega)
	num = TP * TN - FP * FN
	denom = (TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)
	if denom == 0:
		return 0
	#print(TP, TN, FP, FN)
	#print(denom)
	return num / np.sqrt(denom)