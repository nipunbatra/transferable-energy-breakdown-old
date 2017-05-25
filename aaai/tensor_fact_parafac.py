from mf_core import *

import numpy as np
import pandas as pd
import sys
from create_matrix import *
from sklearn.model_selection import LeaveOneOut
from scipy.optimize import nnls

import pickle
from tensorly.decomposition import parafac, non_negative_parafac


APPLIANCES = ['fridge', 'hvac', 'wm', 'mw', 'oven', 'dw']
region = "Austin"
year = 2014

def un_normalize(x, maximum, minimum):
    return (maximum-minimum)*x + minimum


pred = {}
for appliance in APPLIANCES[:]:
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
	#df = (1.0*(df-col_min))/(col_max-col_min)
	tensor = df.values.reshape((len(df), 2, months))
	M, N, O = tensor.shape
	mask = np.ones(M).astype('bool')

	for rank in range(1, 7):
		print rank, appliance
		pred[appliance][rank] = {}
		for i, home in enumerate(df.index[:]):

			tensor_copy = tensor.copy()
			mask_i = mask.copy()
			mask_i[i] = False
			tensor_to_factorise = tensor_copy[mask_i]

			X, Y, Z = non_negative_parafac(tensor_to_factorise, rank)

			assert(X.shape[0]==M-1)
			alpha = np.einsum('nk, ok -> nok', Y, Z).reshape((N * O, rank))
			beta = tensor_copy[~mask_i].reshape(N * O, 1)
			# Learn X_M from aggregate energy values
			X_M = nnls(alpha[:months], beta[:months].reshape(-1, ))[0].reshape((1, rank))
			prediction = np.einsum('ir, jr, kr -> ijk', X_M, Y, Z)
			# Only the appliance energy
			pred_appliance = prediction[0, 1, :]
			pred[appliance][rank][home] = pred_appliance
		pred[appliance][rank] = pd.DataFrame(pred[appliance][rank]).T

pickle.dump(pred, open('predictions/tensor-parafac.pkl', 'w'))
