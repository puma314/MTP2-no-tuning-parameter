import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from utils_plotting_parameters import PlotParams
from utils_evaluation import MCC

plot_params = PlotParams()

def get_dataframe(results_dict, metric="MCC"):
	"""Returns dataframe from results_dict."""
	tuples = []
	all_algos = set([key[0] for key in results_dict.keys()])
	all_Ns = set([key[1] for key in results_dict.keys()])
	max_replications = max([key[2] for key in results_dict.keys()])

	metrics_dict = {}
	for results_info, (omega, omega_hat) in results_dict.items():
		algo_name, N, repl = results_info
		assert metric == "MCC"
		mcc = MCC(omega_hat, omega)
		metrics_dict[results_info] = mcc

	return pd.DataFrame(
		[results_info + (metric,) for results_info, metric in metrics_dict.items()],
		columns = ['Algorithm', 'N', 'Replication', 'MCC'])


def generate_figure_1(results_df, graph_type):
	"""Generates Figure 1 in the paper given a results dictionary.

	Args:
		results_df: Dataframe with results.
	"""
	sns.set_style("whitegrid")
	fig = plt.figure(figsize=(5, 5))
	ax = fig.gca()
	ax.set_yticks(np.linspace(0, 1.0, 6))

	groupby = results_df.groupby('Algorithm')
	x_axis = sorted(results_df.N.unique())
	for algo_name in results_df.Algorithm.unique():
		algo_group = groupby.get_group(algo_name)
		y_axis = []
		y_axis_std = []
		for N in x_axis:
			mean_metric = algo_group.groupby('N').get_group(N).MCC.mean()
			std_metric = algo_group.groupby('N').get_group(N).MCC.std()
			y_axis.append(mean_metric)
			y_axis_std.append(std_metric)
		y_axis = np.array(y_axis)
		y_axis_std = np.array(y_axis_std)
		plt.plot(x_axis, y_axis,
			label=plot_params.labels[algo_name],
			marker=plot_params.markers[algo_name],
			linestyle=plot_params.linestyles[algo_name],
			linewidth=1.5,
			color=plot_params.colors[algo_name])
		plt.fill_between(x_axis, y_axis+y_axis_std, y_axis-y_axis_std,
			facecolor=plot_params.colors[algo_name], alpha=0.2)

	plt.legend(loc='lower right')
	plt.title("{} graph".format(graph_type))
	plt.xlabel("N")
	plt.ylabel("MCC")
	plt.show()
