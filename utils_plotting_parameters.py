# algo_name to displayed name
labels = {
	'algorithm_1': 'Our algorithm', 
	'SH_best': 'SH (best q)', 
	'SH_0.7': 'SH (no stability selection)',
	'SH': 'SH', 
	'nbsel': 'nbsel', 
	'glasso': 'glasso', 
	'CMIT': 'CMIT',
	'algorithm_1_gamma_85':  "Our algorithm: " + r'$\gamma=0.85$',
	'TIGER': 'TIGER',
	'CLIME': 'CLIME',
	'stability_nbsel': 'stability nbsel', 
	'stability_glasso': 'stability glasso', 
	'stability_SH': 'stability SH',
	'stability_CMIT': 'stability CMIT'

}

# algo_name to marker type
markers = {
	'algorithm_1': 'o', 
	'SH_best': '<', 
	'SH_0.7': '<', 
	'SH': '>', 
	'nbsel': '^',
	'glasso': 'v',
	'CMIT': 'd', 
	'algorithm_1_gamma_85': '*', 
	'TIGER': '<', 
	'CLIME': '*',
	'stability_nbsel': '^',
	'stability_glasso': 'v',
	'stability_SH': '>',
	'stability_CMIT': 'd'
}

# algo_name to line-style
linestyles = {
	'algorithm_1': 'solid', 
	'SH_best': 'dashdot', 
	'SH_0.7': 'dashdot', 
	'SH': 'dashdot', 
	'nbsel': 'dotted', 
	'glasso': (0, (3, 1, 1, 1)), 
	'CMIT': 'dashed',
	'algorithm_1_gamma_85': 'solid', 
	'TIGER': 'dotted', 
	'CLIME': 'solid',
	'stability_nbsel': 'dotted',
	'stability_glasso': (0, (3, 1, 1, 1)), 
	'stability_SH': 'dashdot',
	'stability_CMIT': 'dashed'
}

# algo_name to colors
colors = {
    'algorithm_1': 'C0',
    'glasso': 'C1',
    'nbsel': 'C2',
    'CMIT': 'C3',
    'SH': 'C4',
    'SH_best': 'C6',
    'SH_0.7': 'C6',
    'algorithm_1_gamma_85': 'C9',
    'TIGER': 'C8',
    'CLIME': 'C7',
    'stability_nbsel': 'C2',
    'stability_glasso': 'C1',
    'stability_SH': 'C4',
    'stability_CMIT': 'C3'

}

class PlotParams():
	def __init__(self):
		self.labels = labels
		self.markers = markers
		self.colors = colors
		self.linestyles = linestyles