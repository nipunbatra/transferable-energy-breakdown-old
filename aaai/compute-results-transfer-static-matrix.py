from aaai18.common import compute_rmse_fraction

from create_matrix import *

appliance_index = {appliance: APPLIANCES_ORDER.index(appliance) for appliance in APPLIANCES_ORDER}

import os
import pickle
source, target = sys.argv[1:]
cost = 'l21'
out = {}
for train_percentage in range(10, 110, 20):
	out[train_percentage] = {}
	for random_seed in range(5):
		out[train_percentage][random_seed] ={}

		name = "{}-{}-{}-{}".format(source, target, random_seed, train_percentage)
		directory = os.path.expanduser('~/aaai2017/transfer-static-matrix-{}_{}/'.format(source, target))
		if not os.path.exists(directory):
			os.makedirs(directory)
		filename = os.path.expanduser('~/aaai2017/transfer-static-matrix-{}_{}/'.format(source, target) + name + '.pkl')
		try:
			pr = pickle.load(open(filename, 'r'))
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

pickle.dump(out, open("predictions/{}-{}-transfer-cv-static-matrix.pkl".format(source, target),"w"))



