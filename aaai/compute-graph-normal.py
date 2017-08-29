from aaai18.common import compute_rmse_fraction

from create_matrix import *

appliance_index = {appliance: APPLIANCES_ORDER.index(appliance) for appliance in APPLIANCES_ORDER}

import os
import pickle
source = sys.argv[1]
out = {}

for train_percentage in [10, 30, 50, 70,90]:

	out[train_percentage] ={}
	for random_seed in range(5):
		out[train_percentage][random_seed] = {}
		name = "{}-{}".format(random_seed, float(train_percentage))

		directory = os.path.expanduser('~/git/pred_graph/{}/'.format(source))

		filename = os.path.expanduser('~/git/pred_graph/{}/'.format(source) + name + '.pkl')

		try:
			pr = pickle.load(open(filename, 'r'))
			pred = pr['Predictions']
			for appliance in APPLIANCES_ORDER[1:]:
				prediction = pred[appliance]
				if appliance == "hvac":
					prediction = prediction[range(4, 10)]
				out[train_percentage][random_seed][appliance]= \
			compute_rmse_fraction(appliance, prediction, source)[2]
			print("Computed for: {}".format(name))

		except Exception, e:
			print(e)
			print("Exception")

	out[train_percentage] = pd.DataFrame(out[train_percentage]).mean(axis=1)

pickle.dump(out, open("predictions/{}-graph-normal-cv.pkl".format(source),"w"))
