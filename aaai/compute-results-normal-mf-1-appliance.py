from common import  compute_rmse_fraction
import os
import pickle

region = "SanDiego"
target = region
year = 2014
APPLIANCES = ['fridge', 'hvac', 'wm', 'mw', 'oven', 'dw']


for appliance in APPLIANCES:
	for features in ['energy', 'energy_static']:

		for cost in ['absolute']:
			for random_seed in range(5):
				for train_percentage in range(10, 110, 10):


					name = "{}-{}-{}".format(features, random_seed, train_percentage)
					directory = os.path.expanduser('~/aaai2017/Normal-MF_{}/{}'.format(target, appliance))

					filename = '{}'.format(os.path.join(directory, name + '.pkl'))
					out = pickle.load(open(filename, 'r'))
					pred = out['Prediction']

