from aaai18.common import compute_rmse_fraction

from create_matrix import *

appliance_index = {appliance: APPLIANCES_ORDER.index(appliance) for appliance in APPLIANCES_ORDER}

import os
import pickle
source = 'SanDiego'
cost = 'l21'
out = {}
for static_fac in ['None']:
	out[static_fac] = {}
	for lam in [0]:
		out[static_fac][lam] = {}
		for train_percentage in range(10, 110, 10):

			out[static_fac][lam][train_percentage] ={}
			for random_seed in range(5):
				out[static_fac][lam][train_percentage][random_seed] = {}
				name = "{}-{}-{}-{}-{}-{}".format(source, static_fac, lam*1.0, random_seed, train_percentage*1., cost)

				directory = os.path.expanduser('~/aaai2017/normal_{}_{}/'.format(source, cost))

				filename = os.path.expanduser('~/aaai2017/normal_{}_{}/'.format(source, cost) + name + '.pkl')

				try:
					pr = pickle.load(open(filename, 'r'))
					pred = pr['Predictions']
					for appliance in APPLIANCES_ORDER[1:]:
						prediction = pred[appliance]
						if appliance == "hvac":
							prediction = prediction[range(4, 10)]
						out[static_fac][lam][train_percentage][random_seed][appliance]= \
					compute_rmse_fraction(appliance, prediction, source)[2]
					print("Computed for: {}".format(name))

				except Exception, e:
					print(e)
					print("Exception")

			out[static_fac][lam][train_percentage] = pd.DataFrame(out[static_fac][lam][train_percentage]).mean(axis=1)

pickle.dump(out, open("predictions/{}-sparse-normal-cv.pkl".format(source),"w"))



