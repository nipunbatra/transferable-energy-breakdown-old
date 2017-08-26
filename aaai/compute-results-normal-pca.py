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
source = 'SanDiego'
cost = 'l21'
out = {}
for static_fac in ['None']:
	out[static_fac] = {}

	out[static_fac] = {}
	for train_percentage in [x*1. for x in range(10, 110, 10)]:

		out[static_fac][train_percentage] ={}
		for random_seed in range(5):
			out[static_fac][train_percentage][random_seed] = {}
			name = "{}-{}-{}-{}-{}".format(source, static_fac, random_seed, train_percentage, cost)
			directory = os.path.expanduser('~/aaai2017/pca-normal_{}_{}/'.format(source, cost))
			if not os.path.exists(directory):
				os.makedirs(directory)
			filename = os.path.expanduser('~/aaai2017/pca-normal_{}_{}/'.format(source, cost) + name + '.pkl')

			try:
				pr = pickle.load(open(filename, 'r'))
				pred = pr['Predictions']
				for appliance in APPLIANCES_ORDER[1:]:
					prediction = pred[appliance]
					if appliance == "hvac":
						prediction = prediction[range(4, 10)]
					out[static_fac][train_percentage][random_seed][appliance]= \
				compute_rmse_fraction(appliance, prediction, source)[2]
				print("Computed for: {}".format(name))

			except Exception, e:
				print(e)
				print("Exception")

		out[static_fac][train_percentage] = pd.DataFrame(out[static_fac][train_percentage]).mean(axis=1)

pickle.dump(out, open("predictions/pca-{}-sparse-normal-cv.pkl".format(source),"w"))



