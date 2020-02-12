from multiprocessing import Pool
from collections import namedtuple
import sys
import running_wrappers
import pickle

NUM_CORES = 6
NUM_ITERS = 20

GraphParams = namedtuple('GraphParams', 'N eta p d ratios')
AlgoParams = namedtuple('AlgoParams', 'stability_samples M pi')

# new_grid_ratios = [0.5, 0.85, 1., 1.25, 1.5, 2]

# grid_3D_old_ratios = [r/524. for r in [200, 250, 300, 400, 500]]
# grid_old_ratios = [r/529. for r in [75, 100, 150, 200, 250]]

graph_params_dict = {
    'chain': GraphParams(p=100, N=[25, 50, 100, 200], eta=1, ratios=None, d=None), #p, N, eta
    'star': GraphParams(p=100, d=[10, 20, 30, 50], N=50, eta=1, ratios=None), #p, d, N, eta
    'random': GraphParams(p=100, d=0.01, ratios=[r/100. for r in [25, 50, 100, 200]], eta=1, N=None), #p, d, ratio over 500, eta
    #'grid_3D': GraphParams(p=4, ratios=new_grid_ratios, eta=2, N=None, d=None), #p, ratio over 524, eta
    'grid': GraphParams(p=10, ratios=[r/100. for r in [25,50,100,200]], eta=1, N=None, d=None) #p, ratio over 529, eta
}

# graph_params_dict = {
#     'chain': GraphParams(p=20, N=[20, 25, 30, 35, 40], eta=1, ratios=None, d=None), #p, N, eta
#     'star': GraphParams(p=20, d=[10, 15, 20, 25, 30], N=50, eta=1, ratios=None), #p, d, N, eta
#     'random': GraphParams(p=20, d=0.01, ratios=[r/500. for r in [300, 375, 500, 750, 1000]], eta=1, N=None), #p, d, ratio over 500, eta
#     'grid_3D': GraphParams(p=2, ratios=[r/524. for r in [200, 250, 300, 400, 500]], eta=2, N=None, d=None), #p, ratio over 524, eta
#     'grid': GraphParams(p=3, ratios=[r/529. for r in [75, 100, 150, 200, 250]], eta=2, N=None, d=None) #p, ratio over 529, eta
# }


algo_params = AlgoParams(stability_samples=50, M=7./9., pi=0.8)
run_name = 'DO_p_100'

with open("{}_algo_params.pkl".format(run_name), 'wb') as f:
	pickle.dump(algo_params, f)

with open("{}_graph_params_dict.pkl".format(run_name), 'wb') as f:
	pickle.dump(graph_params_dict, f)


if __name__ == "__main__":
	# graph_type, num_iters = sys.argv[1], int(sys.argv[2])
	# assert graph_type in ['chain', 'star', 'random', 'grid_3D', 'grid']
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
		for graph_type in ['chain']:
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

	with Pool(NUM_CORES) as p:
		p.map(run_num_param_wrapper, params)




