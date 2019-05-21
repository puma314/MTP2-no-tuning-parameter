from multiprocessing import Pool
from collections import namedtuple, defaultdict
import pickle
import numpy as np
from paper_sims_util import confusion, random_graph
from final_algo import GET_ALGOS_ROC
import random, sys, time

NUM_CORES = 6
NUM_GRAPHS = 2

algo_lambdas = {
	'our': np.linspace(0.75, 0.95, num=2),
	'SH': np.linspace(0.05, 1., num=2),
	'anand': [(1,x) for x in np.logspace(-4,1.2, num=2)], #+ [(2,x) for x in np.logspace(-4,1.2, num=10)],
	# [(1, 0.0001), (1, 0.0005), 
	# 			(1,0.001), (1,0.01), 
	# 			(1,0.1), (1,0.5), 
	# 			(1,1.0), (1,2.0)],
	'nbsel': np.logspace(-6, 1.2, num=2),#[0.000001, 0.0001, 0.001, 0.01, 0.1, 0.5, 1.0, 2.0],
	'glasso': np.logspace(-6, 1.2, num=2)#[0.0001, 0.001, 0.01, 0.1, 0.5, 1.0, 2.0]
}

ALL_ALGOS = GET_ALGOS_ROC()

if __name__ == "__main__":
	# _, N, p, d = sys.argv
	# N = int(N)
	# p = int(p)
	# d = float(d)
	# print(N, p, d)

	N = int(sys.argv[1])
	p = 10
	d = 0.01

	print("Working on N={} d={}".format(N, d))
	start = time.time()
	run_name = 'TEST_ROC'.format(N, p,d)
	RandomGraphParams = namedtuple('RandomGraphParams', 'N p d')
	random_graph_params = RandomGraphParams(N=N, p=p, d=d)

	with open("{}_random_graph_params.pkl".format(run_name), 'wb') as f:
		pickle.dump(random_graph_params, f)

	with open("{}_algo_lambdas.pkl".format(run_name), 'wb') as f:
		pickle.dump(algo_lambdas, f)

	def run_graph_num(graph_num):
		print('Working on {}'.format(graph_num))
		omega = random_graph(random_graph_params.p, random_graph_params.d)
		sigma = np.linalg.inv(omega)
		np.random.seed(random.SystemRandom().randint(0, 2**32-2))
		X = np.random.multivariate_normal(mean = np.zeros(omega.shape[0]), 
			cov = np.linalg.inv(omega), 
			size = random_graph_params.N)

		result = defaultdict(dict)
		TPFP = defaultdict(dict)
		reconstruction_info = {}
		for algo_name, lambdas in algo_lambdas.items():
			algo = ALL_ALGOS[algo_name]
			if algo_name in ['SH', 'anand']:
				all_lambdas_res, recon_info = algo(X, lambdas)
				reconstruction_info[algo_name] = recon_info
				for l, omega_hat in all_lambdas_res.items():
					result[algo_name][l] = omega_hat
					if omega_hat is None:
						continue
					TP, TN, FP, FN = confusion(omega_hat, omega)
					TPR = TP/(TP + FN)
					FPR = FP/(FP + TN)
					TPFP[algo_name][l] = (FPR, TPR)
			else:
				for lamb in lambdas:
					omega_hat = algo(X, lamb)
					result[algo_name][lamb] = omega_hat
					if omega_hat is None:
						continue
					TP, TN, FP, FN = confusion(omega_hat, omega)
					TPR = TP/(TP + FN)
					FPR = FP/(FP + TN)
					TPFP[algo_name][lamb] = (FPR, TPR)
		
		run_result = (result, TPFP, omega, X, reconstruction_info)

		with open('{}_{}_result.pkl'.format(run_name, graph_num), 'wb') as f:
			pickle.dump(run_result, f)

	with Pool(NUM_CORES) as pool:
		pool.map(run_graph_num, range(NUM_GRAPHS))

	end = time.time()
	print('Took ',end-start, 'seconds')




