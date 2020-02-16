import comparison_algorithms
from collections import defaultdict
from main_algorithm import get_batch
import numpy as np

def stability_selection(algo, X, algo_params_dict, NUM_SUBSAMPLES=10):
	N, p = X.shape
	edges = []
	for i in range(p):
		for j in range(i+1, p):
			edges.append((i,j))
	subN = N//2
	results = {}
	probs = {}
	param_names = list(algo_params_dict.keys())
	num_params = len(algo_params_dict[param_names[0]])
	for n in range(num_params):
		param_values = tuple([algo_params_dict[param_name][n] for param_name in param_names])
		algo_params = dict(zip(param_names, param_values))
		results[param_values] = []
		for _ in range(NUM_SUBSAMPLES):
			batch = get_batch(X, N//2)
			res = algo(batch, **algo_params)
			results[param_values].append(res)
			if res is None:
				break
		if results[param_values][-1] is not None:
			probs[param_values] = defaultdict(int)
			for res in results[param_values]:
				if res is None:
					continue
				for e in edges:
					e_val = res[e]
					if e_val != 0.:
						probs[param_values][e] += 1
			for e in edges:
				probs[param_values][e] /= len(results[param_values])
	return results, probs

def get_stability_edges(probs, pi):
	first_key = list(probs.keys())[0]
	edges = list(probs[first_key].keys())
	edges_plot = defaultdict(list)
	param_values = probs.keys()
	for e in edges:
		for param_value in param_values:
			edges_plot[e].append(probs[param_value][e])
	true_edges = set()
	for e, pis in edges_plot.items():
		if max(pis) >= pi:
			true_edges.add(e)
	return true_edges

def stability_wrapper(algo, num_subsamples, pi):
	# Given an algorithm, returns a stability selection version of it.
	def f(X, algo_params_dict):
		results, probs = stability_selection(
			algo, X, algo_params_dict, num_subsamples)
		res = get_stability_edges(probs, pi)
		n, p = X.shape
		omega = np.zeros((p,p))
		for e in res:
			omega[e] = 1
			omega[e[::-1]] = 1
		return omega  # results, probs
	return f

def stability_nbsel(num_subsamples, pi):
	return stability_wrapper(
		comparison_algorithms.nbsel, num_subsamples, pi)

def stability_glasso(num_subsamples, pi):
	return stability_wrapper(
		comparison_algorithms.glasso, num_subsamples, pi)

def stability_CMIT(num_subsamples, pi):
	return stability_wrapper(
		comparison_algorithms.CMIT, num_subsamples, pi)

def stability_SH(num_subsamples, pi):
	def f(X, algo_params_dict):
		assert 'q' in algo_params_dict
		thresholds = algo_params_dict['q']
		N, p = X.shape
		MTP2_precs = []
		subN = N//2
		for _ in range(num_subsamples):
			batch = get_batch(X, subN)
			MTP2res = comparison_algorithms.run_single_MTP(np.cov(batch.T))
			MTP2_precs.append(MTP2res)
		
		edges = []
		for i in range(p):
			for j in range(i+1, p):
				edges.append((i,j))
		
		results = defaultdict(list)
		probs = {}
		for thres in algo_params_dict['q']:
			for prec in MTP2_precs:
				results[thres].append(comparison_algorithms.attr_threshold_new(prec, thres))

			probs[thres] = defaultdict(int)
			for res in results[thres]:
				for e in edges:
					e_val = res[e]
					if e_val != 0:
						probs[thres][e] += 1
			for e in edges:
				probs[thres][e] /= len(results[thres])

		stable_edges = get_stability_edges(probs, pi)
		omega = np.zeros((p,p))
		for e in stable_edges:
			omega[e] = 1
			omega[e[::-1]] = 1
		return omega
	return f

