from multiprocessing import Pool
from collections import namedtuple
import sys
import running_wrappers
import pickle

NUM_CORES = 6
NUM_ITERS = 1

GraphParams = namedtuple('GraphParams', 'N eta p d ratios')
AlgoParams = namedtuple('AlgoParams', 'stability_samples M pi')

graph_params_dict = {
    'chain': GraphParams(p=10, N=[25, 50], eta=1, ratios=None, d=None), #p, N, eta
    'star': GraphParams(p=10, d=[10, 20], N=50, eta=1, ratios=None), #p, d, N, eta
    'random': GraphParams(p=10, d=0.01, ratios=[r/100. for r in [25, 50]], eta=1, N=None), #p, d, ratio over 500, eta
    'random_more_dense': GraphParams(p=100, d=0.05, ratios=[r/100. for r in [25, 50]], eta=1, N=None),

    #'grid_3D': GraphParams(p=4, ratios=new_grid_ratios, eta=2, N=None, d=None), #p, ratio over 524, eta
    'grid': GraphParams(p=10, ratios=[r/100. for r in [25,50]], eta=1, N=None, d=None) #p, ratio over 529, eta
}

algo_params = AlgoParams(stability_samples=2, M=7./9., pi=0.8)
run_name = 'TEST_RUN'

with open("{}_algo_params.pkl".format(run_name), 'wb') as f:
	pickle.dump(algo_params, f)

with open("{}_graph_params_dict.pkl".format(run_name), 'wb') as f:
	pickle.dump(graph_params_dict, f)


if __name__ == "__main__":
	#graph_type = sys.argv[1]
	#assert graph_type in ['chain', 'star', 'random', 'random_more_dense', 'grid']
	# wrapper = running_wrappers.WRAPPERS[graph_type]
	# graph_params = graph_params_dict[graph_type]
	# def run_num_wrapper(run_num):
	# 	try:
	# 		wrapper(graph_params, algo_params, run_name, run_num)
	# 	except:
	# 		print("ERROR on {}".format(run_num))

	# with Pool(NUM_CORES) as p:
	# 	p.map(run_num_wrapper, range(num_iters))

	params = []
	for run_num in range(NUM_ITERS):
		for graph_type in ['chain', 'star', 'random', 'random_more_dense', 'grid']:
			params.append((graph_type, run_num))

	def run_num_param_wrapper(args):
		graph_type, run_num = args
		print('Running on {} for {}'.format(graph_type, run_num))
		wrapper = running_wrappers.WRAPPERS[graph_type]
		graph_params = graph_params_dict[graph_type]
		try:
			wrapper(graph_params, algo_params, run_name, run_num)
		except:
			print(sys.exc_info()[0])
			print("ERROR on {}".format(run_num))

	print(params)
	with Pool(NUM_CORES) as p:
		p.map(run_num_param_wrapper, params)




