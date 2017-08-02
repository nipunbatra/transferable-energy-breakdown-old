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
out = {}
for static_fac in ['None','static']:
	out[static_fac] = {}
	for lam in [0.001, 0.01, 0.1, 1.]:
		out[static_fac][lam] = {}
		for num_home_factors in range(3, 9):
			out[static_fac][lam][num_home_factors] = {}
			for num_season_factors in range(1, 9):
				out[static_fac][lam][num_home_factors][num_season_factors] = {}
				for train_percentage in [x*1. for x in range(10, 110, 10)]:
					out[static_fac][lam][num_home_factors][num_season_factors][int(train_percentage)] = {}
					for random_seed in range(5):
						name = "{}-{}-{}-{}-{}-{}".format(static_fac, lam, num_home_factors, num_season_factors,
						                                  random_seed, train_percentage)
						print(name)
						pr = pickle.load(open(os.path.expanduser('~/aaai2017/transfer/' + name + '.pkl'), 'r'))
						o = {}
						for appliance in APPLIANCES_ORDER:
							if appliance=="hvac":
								o[appliance] = compute_rmse_fraction(appliance, pr[appliance][range(4, 10)], 'SanDiego')[2]
							else:
								o[appliance] = compute_rmse_fraction(appliance, pr[appliance], 'SanDiego')[2]
						out[static_fac][lam][num_home_factors][num_season_factors][int(train_percentage)][random_seed] = o




