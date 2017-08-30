"""
Run all the code on HCDM

"""

import os
import pickle
import pandas as pd
from common import APPLIANCES_ORDER, compute_rmse_fraction
source = "Austin"
target = "Boulder"

out = {}
params = {}
for case in [2, 4]:
	out[case] = {}
	params[case] = {}
	for constant_use in ['True', 'False']:
		out[case][constant_use] = {}
		params[case][constant_use] = {}
		for static_use in ['True', 'False']:
			out[case][constant_use][static_use] = {}
			params[case][constant_use][static_use] = {}
			for setting in ['normal','transfer']:
				out[case][constant_use][static_use][setting] = {}
				params[case][constant_use][static_use][setting] = {}
				for train_percentage in [10., 30.,
				                         50.,  70.,  90.]:
					out[case][constant_use][static_use][setting][train_percentage] = {}
					params[case][constant_use][static_use][setting][train_percentage] = {}
					for random_seed in range(4):
						out[case][constant_use][static_use][setting][train_percentage]
						params[case][constant_use][static_use][setting][train_percentage]
						if setting == "transfer":
							name = "{}-{}-{}-{}".format(source, target, random_seed, train_percentage)
						else:
							name = "{}-{}-{}".format(target, random_seed, train_percentage)

						directory = os.path.expanduser(
							'~/git/scalable-nilm/aaai18/predictions/TF/{}/case-{}/{}/{}'.format(setting, case, static_use, constant_use))
						if not os.path.exists(directory):
							os.makedirs(directory)
						filename = os.path.join(directory, name + '.pkl')
						try:
							out[case][constant_use][static_use][setting][train_percentage][random_seed]={}
							params[case][constant_use][static_use][setting][train_percentage][random_seed] = {}
							pr = pickle.load(open(filename, 'r'))
							pred = pr['Predictions']
							parameter_data = pr['Learning Params']
							params[case][constant_use][static_use][setting][train_percentage][random_seed] = parameter_data
							for appliance in APPLIANCES_ORDER[1:]:
								prediction = pred[appliance]
								if appliance == "hvac":
									prediction = prediction[range(4, 10)]
								out[case][constant_use][static_use][setting][train_percentage][random_seed][appliance] = \
									compute_rmse_fraction(appliance, prediction, target)[2]
							print("Computed for: {}".format(name))

						except Exception, e:
							print(e)
							print("Exception")
					out[case][constant_use][static_use][setting][train_percentage] = pd.DataFrame(out[case][constant_use][static_use][setting][train_percentage]).mean(axis=1)

pickle.dump(out, open('../predictions/lr-tf-{}-{}.pkl'.format(source, target), 'w'))
pickle.dump(params, open('../predictions/params-lr-tf-{}-{}.pkl'.format(source, target), 'w'))
