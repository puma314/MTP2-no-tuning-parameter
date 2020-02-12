import numpy as np
import random

def is_MTP2(omega):
	p = omega.shape[0]
	for i in range(p):
		for j in range(p):
			if i != j:
				if not np.isclose(omega[i,j], 0, atol = 1e-6):
					assert omega[i,j] <= 0, omega
	return True

def grid_graph(p, mult = 1.05):
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
	omega = mult * delta * np.eye(B.shape[0]) - B
	sigma = np.linalg.inv(omega)
	D = np.diag(np.power(np.diag(sigma), -0.5))
	D_inv = np.linalg.inv(D)
	omega = D_inv.dot(omega).dot(D_inv)
	assert is_MTP2(omega)
	return omega

def grid_3D(p):
	adj = np.zeros((p**3,p**3))
	def n(t):
		return t[0]*p**2 + t[1]*p + t[2]
	def in_bounds(t):
		return min(t) >= 0 and max(t) < p

	for i in range(p):
		for j in range(p):
			for k in range(p):
				new_ts = [
					(i+1, j,k), (i-1,j,k), 
					(i, j-1, k), (i,j+1,k), 
					(i,j,k-1), (i,j,k+1)
				]
				for neigh in new_ts:
					if in_bounds(neigh):
						adj[n((i,j,k)), n(neigh)] = 1
	B = adj
	delta = np.real(sorted(np.linalg.eigvals(B))[-1])
	omega = 1.05 * delta * np.eye(B.shape[0]) - B
	sigma = np.linalg.inv(omega)
	D = np.diag(np.power(np.diag(sigma), -0.5))
	D_inv = np.linalg.inv(D)
	omega = D_inv.dot(omega).dot(D_inv)
	assert is_MTP2(omega)
	return omega

def chain(p):
	sigma = np.zeros((p,p))
	for j in range(p):
		for k in range(p):
			sigma[j,k] = 0.9**(abs(j-k))
	omega = np.linalg.inv(sigma)
	assert is_MTP2(omega)
	return omega

def star(d, p):
	rho = 0.6 / np.power(d, 0.25) * np.hstack((np.ones(d), np.zeros(p-d-1)))
	D = 1 + np.sum(np.square(rho))
	row_0 = np.hstack(((D), -rho))
	rho_2d = np.expand_dims(rho, 1)
	rest = np.hstack((-rho_2d, np.eye(p-1)))
	omega = np.vstack((row_0, rest))
	assert is_MTP2(omega)
	return omega

def decay(p):
	omega = np.zeros((p,p))
	for j in range(p):
		for k in range(p):
			if j == k:
				omega[j][k] = 1
			else:
				omega[j][k] = (-1)*np.exp(-abs(j-k) * 6 / 5)
	assert is_MTP2(omega)
	return omega

GRAPH_TYPE_MAP = {
	"grid": grid_graph,
	"grid_3d": grid_3D,
	"chain": chain,
	"star": star,
	"decay": decay
}

def generate_graph(graph_type, **kwargs):
	return GRAPH_TYPE_MAP[graph_type](**kwargs)

def get_samples(omega, N):
	sigma = np.linalg.inv(omega)
	np.random.seed(random.SystemRandom().randint(0, 2**32-2))
	X = np.random.multivariate_normal(mean = np.zeros(omega.shape[0]), 
									  cov = np.linalg.inv(omega), 
									  size = N)
	return X