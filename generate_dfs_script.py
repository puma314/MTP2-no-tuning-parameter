import pickle
import numpy as np
from collections import namedtuple, defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from final_algo import attr_threshold
from paper_sims_util import confusion
import time

p = 100
d = 0.01
for N in [500, 200, 100, 50, 25, 1000]:
	run_name = 'ROC_N_{}_p_{}_d_{}'.format(N, p, d)
	NUM_GRAPHS = 30

	all_results = []
	data = []
	for graph_num in range(NUM_GRAPHS):
		print(graph_num)
		try:
			res = pickle.load(open('{}_{}_result.pkl'.format(run_name, graph_num), 'rb'))
		except:
			continue
		result, FPTP_dict, omega, _, recon_info = res
		all_results.append(FPTP_dict)
		
		for algo_name in FPTP_dict.keys():
			for lamb in FPTP_dict[algo_name].keys():
				omega_hat = result[algo_name][lamb]
				TP, TN, FP, FN = confusion(omega_hat, omega)
				#print(TP, TN, FP, FN)
				if TP + FN == 0:
					TPR = 0
				else:
					TPR = TP/(TP + FN)
				if FP+TN == 0:
					FPR = 0
				else:
					FPR = FP/(FP + TN)
				key_lamb = (tuple((lamb,)) if type(lamb) != tuple else lamb)
				data.append((graph_num, algo_name, key_lamb, FPR, TPR, TP, TN, FP, FN))
		with open('{}_data_{}.pkl'.format(run_name, graph_num), 'wb') as f:
			pickle.dump(data, f)
	
	with open('{}_data.pkl'.format(run_name), 'wb') as f:
		pickle.dump(data, f)