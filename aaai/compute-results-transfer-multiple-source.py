from create_matrix import create_matrix_region_appliance_year
from subprocess import Popen
from common import compute_rmse_fraction
from create_matrix import *

from tensor_custom_core import *
from create_matrix import *
from tensor_custom_core import *
from degree_days import dds
appliance_index = {appliance: APPLIANCES_ORDER.index(appliance) for appliance in APPLIANCES_ORDER}

import os
import pickle
source_1, source_2, target = "SanDiego", "Boulder", "Austin"
static_fac ='None'
lam = 0
cost = 'l21'
out = {}
betas = {}
for train_percentage in range(10, 110, 10):
	out[train_percentage] = {}
	betas[train_percentage] = {}
	for random_seed in range(5):
		out[train_percentage][random_seed] ={}


		name = "{}-{}-{}-{}-{}-{}-{}".format(source_1, source_2, target, static_fac, lam, random_seed, train_percentage)
		directory = os.path.expanduser('~/aaai2017/transfer_{}_{}_{}/'.format(source_1, source_2, target))
		if not os.path.exists(directory):
			os.makedirs(directory)
		filename = os.path.expanduser(
			'~/aaai2017/transfer_{}_{}_{}/'.format(source_1, source_2, target) + name + '.pkl')
		try:
			pr = pickle.load(open(filename, 'r'))
			betas[train_percentage][random_seed] = [pr['Learning Params'][x]['Ratio'] for x in pr['Learning Params'].keys()]
			pred = pr['Predictions']
			for appliance in APPLIANCES_ORDER[1:]:
				prediction = pred[appliance]
				if appliance == "hvac":
					prediction = prediction[range(4, 10)]
				out[train_percentage][random_seed][appliance]= \
			compute_rmse_fraction(appliance, prediction, target)[2]
			print("Computed for: {}".format(name))

		except Exception, e:
			print(e)
			print("Exception")

	out[train_percentage] = pd.DataFrame(out[train_percentage]).mean(axis=1)

pickle.dump(out, open("predictions/transfer-multiple-{}-{}-{}.pkl".format(source_1, source_2, target),"w"))



