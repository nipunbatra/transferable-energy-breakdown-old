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
source = sys.argv[1]
out = {}

for train_percentage in [6, 8, 10,12, 14, 16,18,20 ,30,40,  50, 60, 70,80, 90,100]:

	out[train_percentage] ={}
	for random_seed in range(5):
		out[train_percentage][random_seed] = {}
		name = "{}-{}".format(random_seed, float(train_percentage))

		directory = os.path.expanduser('~/git/pred_tf/{}/'.format(source))

		filename = os.path.expanduser('~/git/pred_tf/{}/'.format(source) + name + '.pkl')

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

pickle.dump(out, open("predictions/{}-tf-normal-cv.pkl".format(source),"w"))