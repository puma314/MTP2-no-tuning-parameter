"""This is the code to generate Figure 1.

Run code below with different parameters to generate Figure 1.
in our paper.
"""

from main_algorithm import no_tuning_parameters  # This is our algorithm.
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
	"algorithm_1": no_tuning_parameters
	# nbsel, glasso excluded for now
}
algorithm_parameters = {
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
