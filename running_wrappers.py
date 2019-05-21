from final_algo import GET_ALGOS, get_algo_lambdas
import pickle
import numpy as np
from paper_sims_util import MCC, grid_graph, grid_3D, chain, star, random_graph, is_MTP2
from collections import defaultdict
import os, sys, time
import random

def grid_3D_wrapper(graph_params, algo_params, run_name, run_id):
	p = graph_params.p
	dim = p**3
	Ns = [int(r*dim) for r in graph_params.ratios]
	if min(Ns) < 10:
		print("Ns for grid_3D are too small")
		print(dim, graph_params.ratios)
		print(Ns)
		assert False
	for r, N in zip(graph_params.ratios, Ns):
		omega = grid_3D(p)
		results = get_results(graph_params, algo_params, omega, N)
		fname = "{}_{}_{}_{}_results.pkl".format(run_name, run_id, 'grid_3D', r)
		with open(fname, 'wb') as f:
			pickle_obj = (omega, results)
			pickle.dump(pickle_obj, f)

def grid_3D_loader(graph_params, algo_params, run_name, run_ids):
	p = graph_params.p
	dim = p**3
	Ns = [int(r*dim) for r in graph_params.ratios]
	loaded = defaultdict(list)
	for r, N in zip(graph_params.ratios, Ns):
		for run_id in run_ids:
			fname = "{}_{}_{}_{}_results.pkl".format(run_name, run_id, 'grid_3D', r)
			if not os.path.isfile(fname):
				continue
			with open(fname, 'rb') as f:
				loaded[N].append(pickle.load(f))
	return loaded

def grid_wrapper(graph_params, algo_params, run_name, run_id):
	p = graph_params.p
	dim = p**2
	Ns = [int(r*dim) for r in graph_params.ratios]
	if min(Ns) < 10:
		print("Ns for grid_3D are too small")
		print(dim, graph_params.ratios)
		print(Ns)
		assert False
	for r, N in zip(graph_params.ratios, Ns):
		omega = grid_graph(p)
		results = get_results(graph_params, algo_params, omega, N)
		fname = "{}_{}_{}_{}_results.pkl".format(run_name, run_id, 'grid', r)
		with open(fname, 'wb') as f:
			pickle_obj = (omega, results)
			pickle.dump(pickle_obj, f)

def grid_loader(graph_params, algo_params, run_name, run_ids):
	p = graph_params.p
	dim = p**2
	Ns = [int(r*dim) for r in graph_params.ratios]
	loaded = defaultdict(list)
	for r, N in zip(graph_params.ratios, Ns):
		for run_id in run_ids:
			fname = "{}_{}_{}_{}_results.pkl".format(run_name, run_id, 'grid', r)
			if not os.path.isfile(fname):
				continue
			with open(fname, 'rb') as f:
				loaded[N].append(pickle.load(f))
	return loaded

def chain_wrapper(graph_params, algo_params, run_name, run_id):
	p = graph_params.p
	for N in graph_params.N:
		omega = chain(p)
		results = get_results(graph_params, algo_params, omega, N)
		fname = "{}_{}_{}_{}_results.pkl".format(run_name, run_id, 'chain', N)
		with open(fname, 'wb') as f:
			pickle_obj = (omega, results)
			pickle.dump(pickle_obj, f)

def chain_loader(graph_params, algo_params, run_name, run_ids):
	p = graph_params.p

	loaded = defaultdict(list)
	for N in graph_params.N:
		for run_id in run_ids:
			fname = "{}_{}_{}_{}_results.pkl".format(run_name, run_id, 'chain', N)
			if not os.path.isfile(fname):
				continue
			with open(fname, 'rb') as f:
				print(fname)
				loaded[N].append(pickle.load(f))
	return loaded

def random_wrapper(graph_params, algo_params, run_name, run_id):
	p = graph_params.p
	d = graph_params.d
	dim = p
	Ns = [int(r*dim) for r in graph_params.ratios]
	if max(Ns) < 10:
		Ns = [x*10 for x in Ns]
	for r, N in zip(graph_params.ratios, Ns):
		omega = random_graph(p,d)
		inv_exists = False
		while not inv_exists:
			try:
				sigma = np.linalg.inv(omega)
				inv_exists = True
			except:
				omega = random_graph(p,d)
		results = get_results(graph_params, algo_params, omega, N)
		fname = "{}_{}_{}_{}_results.pkl".format(run_name, run_id, 'random', r)
		with open(fname, 'wb') as f:
			pickle_obj = (omega, results)
			pickle.dump(pickle_obj, f)

def random_loader(graph_params, algo_params, run_name, run_ids):
	p = graph_params.p
	d = graph_params.d
	dim = p
	Ns = [int(r*dim) for r in graph_params.ratios]
	if max(Ns) < 10:
		Ns = [x*10 for x in Ns]
	loaded = defaultdict(list)
	for r, N in zip(graph_params.ratios, Ns):
		for run_id in run_ids:
			fname = "{}_{}_{}_{}_results.pkl".format(run_name, run_id, 'random', r)
			if not os.path.isfile(fname):
				continue
			with open(fname, 'rb') as f:
				loaded[N].append(pickle.load(f))
	return loaded

def star_wrapper(graph_params, algo_params, run_name, run_id):
	p = graph_params.p
	for d in graph_params.d:
		omega = star(d, p)
		results = get_results(graph_params, algo_params, omega, graph_params.N)
		fname = "{}_{}_{}_{}_results.pkl".format(run_name, run_id, 'star', d)
		with open(fname, 'wb') as f:
			pickle_obj = (omega, results)
			pickle.dump(pickle_obj, f)

def star_loader(graph_params, algo_params, run_name, run_ids):
	p = graph_params.p

	loaded = defaultdict(list)
	for d in graph_params.d:
		for run_id in run_ids:
			fname = "{}_{}_{}_{}_results.pkl".format(run_name, run_id, 'star', d)
			if not os.path.isfile(fname):
				continue
			with open(fname, 'rb') as f:
				loaded[d].append(pickle.load(f))
	return loaded

WRAPPERS = {
	'grid_3D': grid_3D_wrapper,
	'grid': grid_wrapper,
	'chain': chain_wrapper,
	'random': random_wrapper,
	'star': star_wrapper
}

LOADERS = {
	'grid_3D': grid_3D_loader,
	'chain': chain_loader,
	'star': star_loader,
	'grid': grid_loader,
	'random': random_loader
}

def get_loaders():
	return LOADERS

def get_results(graph_params, algo_params, omega, N):
	algos = GET_ALGOS(algo_params.stability_samples)
	is_MTP2(omega)
	algo_lambdas = get_algo_lambdas(algo_params.M, graph_params.eta, N, graph_params.p)
	sigma = np.linalg.inv(omega)
	np.random.seed(random.SystemRandom().randint(0, 2**32-2))
	X = np.random.multivariate_normal(mean = np.zeros(omega.shape[0]), 
									  cov = np.linalg.inv(omega), 
									  size = N)
	
	ALL_RESULTS = {}
	for algo_name, algo in algos.items():
		print("Currently on {}".format(algo_name))
		lambdas = algo_lambdas[algo_name]
		if algo_name == 'our':
			our_algo_res = algo(X, algo_params.M)
			results = (our_algo_res, None, None)
		else:
			results = algo(X, lambdas, algo_params.pi)
		ALL_RESULTS[algo_name] = results
		omega_hat = results[0]
		res_mcc = MCC(omega_hat, omega)
		print("Algorithm {} got {} MCC".format(algo_name, res_mcc))
	return ALL_RESULTS
