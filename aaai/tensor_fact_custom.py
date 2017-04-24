from mf_core import *

import numpy as np
import pandas as pd
import sys
from create_matrix import *
from sklearn.model_selection import LeaveOneOut
from scipy.optimize import nnls

import pickle
from tensorly.decomposition import parafac, non_negative_parafac
from tensor_custom_core import *


APPLIANCES = ['fridge', 'hvac', 'wm', 'mw', 'oven', 'dw']
region = "Austin"
year = 2014

def un_normalize(x, maximum, minimum):
    return (maximum-minimum)*x + minimum

a, b = 2, 3


pred = {}
for appliance in APPLIANCES[:]:
	pred[appliance] = {}
	if appliance == "hvac":
		start, stop = 5, 11
	else:
		start, stop = 1, 13
	months = stop - start
	pred[appliance] = {}
	appliance_df = create_matrix_all_entries(region, year, appliance)
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
	M, N, O = tensor.shape
	mask = np.ones(M).astype('bool')

	for case in range(1, 5):
		pred[appliance][case] = {}
		for i, home in enumerate(df.index[:]):


			tensor_copy = tensor.copy()
			mask_i = mask.copy()
			mask_i[i] = False
			tensor_to_factorise = tensor_copy[mask_i]

			H, A, T = learn_HAT(case, tensor_to_factorise, a, b, num_iter=2000, lr=0.1, dis=False)
			prediction = multiply_case(H, A, T, case)
			pred_appliance = un_normalize(prediction[0, 1, :], col_max, col_min)
			pred[appliance][case][home] = pred_appliance
			print appliance, case, i, home, pred_appliance
		pred[appliance][case] = pd.DataFrame(pred[appliance][case]).T

pickle.dump(pred, open('predictions/tensor-custom.pkl', 'w'))
