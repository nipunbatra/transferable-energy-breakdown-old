from mf_core import *

import numpy as np
import pandas as pd
import sys
from create_matrix import *
from sklearn.model_selection import LeaveOneOut
from metric_learn import MLKR
from sklearn.neighbors import KNeighborsRegressor
import pickle

APPLIANCES = ['fridge', 'hvac', 'wm', 'mw', 'oven', 'dw']
region = "Austin"
year = 2014


pred = {}
for appliance in APPLIANCES:
	if appliance == "hvac":
		start, stop = 5, 11
	else:
		start, stop = 1, 13
	pred[appliance] = {}
	appliance_df = create_matrix_all_entries(region, year, appliance)
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

	for features in ['energy', 'energy_static']:
		pred[appliance][features] = {}
		if features == "energy":
			cols = aggregate_cols
		else:
			cols = X_cols

		X_matrix, X_normalised, matrix_max, matrix_min, appliance_cols, aggregate_cols = preprocess(df, dfc, appliance,
		                                                                                            col_max,
		                                                                                            col_min, False)
		static_features = get_static_features(dfc, X_normalised)
		if features=="energy":
			feature_comb = ['None']
		else:
			feature_comb =['occ','area','rooms']
		idx_user, data_user = prepare_known_features(feature_comb, static_features, X_normalised)

		for cost in ['absolute','relative']:
			pred[appliance][features][cost] = {}
			for latent_factors in range(3, 9):
				pred[appliance][features][cost][latent_factors] = {}

				print latent_factors, features, appliance, cost
				loo = LeaveOneOut()
				for train_ix, test_ix in loo.split(appliance_df):
					test_home = appliance_df.index.values[test_ix][0]
					pred[appliance][features][cost][latent_factors][test_home] = {}
					A = create_matrix_factorised(appliance, [test_home], X_normalised)
					X, Y, res = nmf_features(A=A, k=latent_factors, constant=0.01, regularisation=False,
					                         idx_user=idx_user, data_user=data_user,
					                         idx_item=None, data_item=None, MAX_ITERS=10, cost=cost)
					pred_df = create_prediction(test_home, X, Y, X_normalised, appliance,
					                            col_max, col_min, appliance_cols)
					pred[appliance][features][cost][latent_factors][test_home] = pred_df
				pred[appliance][features][cost][latent_factors] = pd.DataFrame(pred[appliance][features][cost][latent_factors]).T

pickle.dump(pred, open('predictions/mf.pkl', 'w'))
