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
source = 'Austin'
target = 'SanDiego'
cost = 'l21'
out = {}
for static_fac in ['None','static']:
	out[static_fac] = {}
	for lam in [0]:
		out[static_fac][lam] = {}
		for train_percentage in range(10, 110, 20):

			out[static_fac][lam][train_percentage] ={}
			for random_seed in range(5):
				name = "{}-{}-{}-{}-{}-{}-{}".format(source, target, static_fac, lam, random_seed, train_percentage,
				                                     cost)
				directory = os.path.expanduser('~/aaai2017/transfer_{}_{}_{}/'.format(source, target, cost))

				filename = os.path.join(directory, name + '.pkl')
				try:
					pr = pickle.load(open(filename, 'r'))
					pred = pr['Prediction']
					'''
										if appliance == "hvac":
						pred = pred[['hvac_{}'.format(month) for month in range(5, 11)]]
						out[static_fac][lam][train_percentage] = \
					compute_rmse_fraction(appliance, pred, target)[2]
					'''

				except:
					pass
				print (name)





