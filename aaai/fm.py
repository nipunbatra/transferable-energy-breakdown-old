from mf_core import *

import numpy as np
import pandas as pd
import sys
from create_matrix import *
from sklearn.model_selection import LeaveOneOut
from metric_learn import MLKR
from sklearn.neighbors import KNeighborsRegressor
import pickle
from fm_core import *

APPLIANCES = ['fridge', 'hvac', 'wm', 'mw', 'oven', 'dw']
region = "Austin"
year = 2014


pred = {}
for appliance in APPLIANCES[:1]:
	if appliance == "hvac":
		start, stop = 5, 11
	else:
		start, stop = 1, 13
	pred[appliance] = {}
	appliance_df = create_matrix_region_appliance_year(region, year, appliance)
	static_cols = ['area', 'total_occupants', 'num_rooms']
	aggregate_cols = [x for x in appliance_df.columns if "aggregate" in x]
	appliance_cols = [x for x in appliance_df.columns if appliance in x]
	energy_cols = np.concatenate([aggregate_cols, appliance_cols])

	df = appliance_df.copy()
	dfc = df.copy()

	df = df[energy_cols]
	col_max = df.max().max()
	col_min = df.min().min()
	df = (1.0*(df-col_min))/(col_max-col_min)
	X_cols = np.concatenate([aggregate_cols, static_cols])

	for features in ['energy_static']:
		pred[appliance][features] = {}
		metadata = dfc[static_cols]
		metadata = metadata.div(metadata.max())
		factorization_machine(df, metadata, col_max, col_min)


#pickle.dump(pred, open('predictions/mf.pkl', 'w'))
