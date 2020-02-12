from multiprocessing import Pool
from collections import namedtuple
import sys
import running_wrappers
import pickle

NUM_CORES = 2
NUM_ITERS = 20

GraphParams = namedtuple('GraphParams', 'N eta p d ratios')
AlgoParams = namedtuple('AlgoParams', 'stability_samples M pi')

graph_params_dict = {
	'star_N_25': GraphParams(p=100, d=[2, 3, 4, 5], N=25, eta=1, ratios=None), #p, d, N, eta
	'star_N_100':GraphParams(p=100, d=[2, 3, 4, 5], N=100, eta=1, ratios=None), #p, d, N, eta
	'star_N_200': GraphParams(p=100, d=[2, 3, 4, 5], N=200, eta=1, ratios=None), #p, d, N, eta
}

algo_params = AlgoParams(stability_samples=50, M=7./9., pi=0.8)




if __name__ == "__main__":
	star_N = sys.argv[1]
	star_N = int(star_N)
	assert star_N in [25, 100, 200]
	run_name = 'DO_p_100_smallstar_{}'.format(star_N)

	with open("{}_algo_params.pkl".format(run_name), 'wb') as f:
		pickle.dump(algo_params, f)

	with open("{}_graph_params_dict.pkl".format(run_name), 'wb') as f:
		pickle.dump(graph_params_dict, f)
		
	params = []
	for run_num in range(NUM_ITERS):
		params.append(('star', run_num))

	def run_num_param_wrapper(args):
		_, run_num = args
		print('Running on {} for {}'.format(run_name, run_num))
		wrapper = running_wrappers.WRAPPERS['star']
		graph_params = graph_params_dict['star_N_{}'.format(star_N)]
		try:
			wrapper(graph_params, algo_params, run_name, run_num)
		except:
			print(sys.exc_info())
			print("ERROR on {}".format(run_num, graph_type))

	print(params)
	with Pool(NUM_CORES) as p:
		p.map(run_num_param_wrapper, params)



