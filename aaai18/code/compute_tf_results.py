"""
Run all the code on HCDM

"""

import os
import pickle
import pandas as pd
from common import APPLIANCES_ORDER, compute_rmse_fraction
source = "Austin"
target = "SanDiego"

out = {}
for case in [2, 4]:
	out[case] = {}
	for static_use in ['True', 'False']:
		out[case][static_use] = {}
		for setting in ['normal','transfer']:
			out[case][static_use][setting] = {}
			for train_percentage in [10., 20., 30., 40.,
			                         50., 60., 70., 80., 90., 100.]:
				out[case][static_use][setting][train_percentage] = {}
				for random_seed in range(5):
					out[case][static_use][setting][train_percentage]
					if setting == "transfer":
						name = "{}-{}-{}-{}".format(source, target, random_seed, train_percentage)
					else:
						name = "{}-{}-{}".format(target, random_seed, train_percentage)

					directory = os.path.expanduser(
						'~/git/scalable-nilm/aaai18/predictions/TF/{}/case-{}/{}'.format(setting, case, static_use))
					if not os.path.exists(directory):
						os.makedirs(directory)
					filename = os.path.join(directory, name + '.pkl')
					try:
						out[case][static_use][setting][train_percentage][random_seed]={}
						pr = pickle.load(open(filename, 'r'))
						pred = pr['Predictions']
						for appliance in APPLIANCES_ORDER[1:]:
							prediction = pred[appliance]
							if appliance == "hvac":
								prediction = prediction[range(4, 10)]
							out[case][static_use][setting][train_percentage][random_seed][appliance] = \
								compute_rmse_fraction(appliance, prediction, target)[2]
						print("Computed for: {}".format(name))

					except Exception, e:
						print(e)
						print("Exception")
				out[case][static_use][setting][train_percentage] = pd.DataFrame(out[case][static_use][setting][train_percentage]).mean(axis=1)

pickle.dump(out, '../predictions/tf-{}-{}.pkl'.format(source, target))