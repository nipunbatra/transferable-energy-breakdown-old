from mf_core import *

import numpy as np
import pandas as pd
import sys
from create_matrix import *
from sklearn.model_selection import LeaveOneOut
from scipy.optimize import nnls

import pickle
from tensor_custom_core import *

APPLIANCES = ['fridge', 'hvac', 'wm', 'mw', 'oven', 'dw']
region = "Austin"
year = 2014


def un_normalize(x, maximum, minimum):
	return (maximum - minimum) * x + minimum


ALL_HOMES = True
ALL_FEATURES = not ALL_HOMES
cost = 'abs'

def tf_factorise_all_appliances(case, a, test_home, cost, all_features):
	df, dfc = create_matrix_single_region(region, year)
	start, stop=1, 13
	energy_cols = np.array([['%s_%d' % (appliance, month) for month in range(start, stop)] for appliance in APPLIANCES_ORDER]).flatten()


	dfc = df.copy()

	df = dfc[energy_cols]
	col_max = df.max().max()
	col_min = df.min().min()
	df = (1.0 * (df - col_min)) / (col_max - col_min)
	tensor = df.values.reshape((len(df), 7, stop-start))
	M, N, O = tensor.shape
	mask = np.ones(M).astype('bool')

	# Find location of test home in the tensor
	i = np.where(df.index.values == test_home)[0][0]
	tensor_copy = tensor.copy()
	tensor_copy[i, 1:, :] = np.NaN

	H, A, T = learn_HAT(case, tensor_copy, a, a, num_iter=2000, lr=0.1, dis=False, cost_function=cost)
	prediction = multiply_case(H, A, T, case)
	# pred_appliance = prediction[i, 1, :]

	pred_appliance = un_normalize(prediction[i, 1, :], col_max, col_min)
	return pred_appliance


def tf_factorise(appliance, case, a, test_home, cost, all_features):
	if appliance == "hvac":
		start, stop = 5, 11
	else:
		start, stop = 1, 13
	months = stop - start
	appliance_df = create_matrix_region_appliance_year(region, year, appliance, all_features)
	aggregate_cols = [x for x in appliance_df.columns if "aggregate" in x]
	appliance_cols = [x for x in appliance_df.columns if appliance in x]
	energy_cols = np.concatenate([aggregate_cols, appliance_cols])

	df = appliance_df.copy()
	dfc = df.copy()

	df = df[energy_cols]
	col_max = df.max().max()
	col_min = df.min().min()
	df = (1.0 * (df - col_min)) / (col_max - col_min)
	tensor = df.values.reshape((len(df), 2, months))
	M, N, O = tensor.shape
	mask = np.ones(M).astype('bool')

	# Find location of test home in the tensor
	i = np.where(df.index.values == test_home)[0][0]
	tensor_copy = tensor.copy()
	tensor_copy[i, 1, :] = np.NaN

	H, A, T = learn_HAT(case, tensor_copy, a, a, num_iter=2000, lr=0.1, dis=False, cost_function=cost)
	prediction = multiply_case(H, A, T, case)
	# pred_appliance = prediction[i, 1, :]

	pred_appliance = un_normalize(prediction[i, 1, :], col_max, col_min)
	return pred_appliance



def main():
	pred = {}
	for appliance in APPLIANCES[:]:
		pred[appliance] = {}
		for case in range(1, 5):

		pred[appliance][case] = {}
		for a in range(1, 6):
			print "*" * 20
			print a, case, appliance
			print "*" * 20

			b = a
			pred[appliance][case][a] = {}
			for i, home in enumerate(df.index[:]):

				pred[appliance][case][a] = pd.DataFrame(pred[appliance][case][a]).T

	if ALL_HOMES:
		pickle.dump(pred, open('predictions/tensor-custom-%s.pkl' % cost, 'w'))
	else:
		pickle.dump(pred, open('predictions/bigger-tensor-custom-%s.pkl' % cost, 'w'))




