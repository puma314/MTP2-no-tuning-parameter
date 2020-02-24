
from utils_plotting import get_dataframe, generate_figure_1
import time
import pickle
import os

output_dir = "chain_test"
results = {}

p = 100
N_list = [25, 50, 100, 200, 500, 1000]
NUM_REPLICATIONS = 20
graph_type = 'chain'

ALL_ALGORITHMS = {
# Tuning parameter free algorithms
	"algorithm_1": None,
	"TIGER": None,
	# "CLIME": CLIME,
# Algorithms with tuning parameters
	# "SH": SH,
	# "glasso": glasso,
	# "nbsel": nbsel,
	# "CMIT": CMIT,
# Stability version of algorithms
	"stability_SH": None,
	"stability_glasso": None,
	"stability_nbsel": None,
	"stability_CMIT": None,
}

for repl in range(NUM_REPLICATIONS):
	for N in N_list:
		for algo_name, _ in ALL_ALGORITHMS.items():
			save_file = os.path.join(output_dir, "{}_{}_{}.pkl".format(algo_name, N, repl))
			if not os.path.exists(save_file):
				continue
			with open(save_file, 'rb') as f:
				omega, omega_hat = pickle.load(f)
			print(algo_name, N, repl)
			results[(algo_name, N, repl)] = (omega, omega_hat)


results_df = get_dataframe(results)
generate_figure_1(results_df, graph_type)