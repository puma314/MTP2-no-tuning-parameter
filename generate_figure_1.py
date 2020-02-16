"""This is the code to generate Figure 1.

Run code below with different parameters to generate Figure 1.
in our paper.
"""

from main_algorithm import no_tuning_parameters  # This is our algorithm.
from comparison_algorithms import CLIME, TIGER, CMIT, nbsel, glasso, SH
from stability_algorithms import stability_nbsel, stability_glasso, stability_CMIT, stability_SH
from utils_graphing import generate_graph, get_samples
from utils_plotting import get_dataframe, generate_figure_1

# Experiment related settings.
p = 10
N_list = [25, 50, 100, 200, 500, 1000]
NUM_REPLICATIONS = 20

# Set which graph you want to plot and what the parameters of the graph are.
graph_type = 'chain'
graph_params = {"p": p}  # Include any other parameters you want.

# Set which algorithms you want to use along with their special parameters
# (if needed).
ALL_ALGORITHMS = {
# Tuning parameter free algorithms
	"algorithm_1": no_tuning_parameters,
	"TIGER": TIGER,
	# "CLIME": CLIME,
# Algorithms with tuning parameters
	# "SH": SH,
	# "glasso": glasso,
	# "nbsel": nbsel,
	# "CMIT": CMIT,
# Stability version of algorithms
	"stability_SH": stability_SH(2, 0.8),
	"stability_glasso": stability_glasso(10, 0.8),
	"stability_nbsel": stability_nbsel(10, 0.8),
	"stability_CMIT": stability_CMIT(10, 0.8),
}

algorithm_parameters = {
	"SH": {"q": 0.8},
	"glasso": {"lamb": 1.},
	"nbsel": {"lamb": 0.1},
	"CMIT": {"xi": 0.1, "eta": 1},
	"stability_SH": {"algo_params_dict": {
		"q": [0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99]
	}},
	"stability_glasso": {"algo_params_dict": {
		"lamb": [0.055, 0.16, 0.45, 1.26, 3.55, 10]
	}},
	"stability_nbsel": {"algo_params_dict": {
		"lamb": [0.055, 0.16, 0.45, 1.26, 3.55, 10]
	}},
	"stability_CMIT": {"algo_params_dict": {
		"xi": [0.055, 0.16, 0.45, 1.26, 3.55, 10],
		"eta": [1, 1, 1, 1, 1, 1]
	}},
}

results = {}

for N in N_list:
	for repl in range(NUM_REPLICATIONS):
		omega = generate_graph(graph_type, **graph_params)
		samples = get_samples(omega, N)
		for algo_name, algo in ALL_ALGORITHMS.items():
			algo_params = algorithm_parameters.get(algo_name, {})
			algorithm_results = algo(samples, **algo_params)
			omega_hat = algorithm_results
			results[(algo_name, N, repl)] = (omega, omega_hat)

results_df = get_dataframe(results)
generate_figure_1(results_df, graph_type)
