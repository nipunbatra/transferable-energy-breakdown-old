import os
import pickle

import pandas as pd

from aaai18.common import compute_rmse_fraction

region = "SanDiego"
target = region
year = 2014
APPLIANCES = ['fridge', 'hvac', 'wm', 'mw', 'oven', 'dw']

cost = 'absolute'
out = {}
for features in ['energy', 'energy_static']:
	out[features] = {}
	for appliance in APPLIANCES:
		out[features][appliance] = {}
		for train_percentage in [x*1. for x in range(10, 110, 10)]:
			out[features][appliance][train_percentage] = {}
			for random_seed in range(5):
				out[features][appliance][train_percentage][random_seed] = {}
				name = "{}-{}-{}".format(features, random_seed, train_percentage)
				directory = os.path.expanduser('~/aaai2017/Normal-MF_{}/{}'.format(target, appliance))
				try:
					filename = '{}'.format(os.path.join(directory, name + '.pkl'))
					p = pickle.load(open(filename, 'r'))
					pred = p['Prediction']
					if appliance=="hvac":
						pred = pred[['hvac_{}'.format(month) for month in range(5, 11)]]
					out[features][appliance][train_percentage][random_seed] = compute_rmse_fraction(appliance, pred, target)[2]
				except Exception, e:
					print(e)
					print(appliance, features, train_percentage, random_seed)
			out[features][appliance][train_percentage] = pd.Series(out[features][appliance][train_percentage][random_seed]).mean()

		import pickle
		pickle.dump(out, open("predictions/mf-baseline-normal.pkl","w"))