from mf_core import *
from tensor_core_tf import *

import numpy as np
import pandas as pd
import sys
from create_matrix import *
from sklearn.model_selection import LeaveOneOut

import pickle

APPLIANCES = ['fridge', 'hvac', 'wm', 'mw', 'oven', 'dw']
region = "Austin"
year = 2014

def un_normalize(x, maximum, minimum):
    return (maximum-minimum)*x + minimum


pred = {}
for appliance in APPLIANCES[2:3]:
	pred[appliance] = {}
	if appliance == "hvac":
		start, stop = 5, 11
	else:
		start, stop = 1, 13
	months = stop - start
	pred[appliance] = {}
	appliance_df = create_matrix_region_appliance_year(region, year, appliance)
	aggregate_cols = [x for x in appliance_df.columns if "aggregate" in x]
	appliance_cols = [x for x in appliance_df.columns if appliance in x]
	energy_cols = np.concatenate([aggregate_cols, appliance_cols])

	df = appliance_df.copy()
	dfc = df.copy()

	df = df[energy_cols]
	col_max = df.max().max()
	col_min = df.min().min()
	df = (1.0*(df-col_min))/(col_max-col_min)
	tensor = df.values.reshape((len(df), 2, months))
	for cost in ['relative','absolute']:
		pred[appliance][cost] = {}
		for h in range(1, 4):
			pred[appliance][cost][h] = {}
			for i, home in enumerate(df.index[:]):
				print i, home, appliance, cost
				print "*" * 20
				tensor_copy = tensor.copy()
				j = 1  # For appliance
				for k, month in enumerate(range(start, stop)):
					tensor_copy[i, j, k] = np.NaN
				H, A, T = tensor_fact(tensor_copy, h, h, cost=cost, max_iter=4000)
				pred_tensor = np.tensordot(np.tensordot(H, A, axes=1), T, axes=1)
				pred[appliance][cost][h][home] = un_normalize(pred_tensor[i, 1, :], col_max, col_min)
			pred[appliance][cost][h]= pd.DataFrame(pred[appliance][cost][h]).T
