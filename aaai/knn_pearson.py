import numpy as np
import pandas as pd
import sys
from create_matrix import *
from sklearn.model_selection import LeaveOneOut
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsRegressor
import pickle

APPLIANCES = ['fridge', 'hvac', 'wm', 'mw', 'oven', 'dw']
region = "Austin"
year = 2014

ALL_HOMES = True
ALL_FEATURES =  not ALL_HOMES

pred = {}
for appliance in APPLIANCES[:]:
	if appliance == "hvac":
		start, stop = 5, 11
	else:
		start, stop = 1, 13
	pred[appliance] = {}
	appliance_df = create_matrix_region_appliance_year(region, year, appliance, all_features=ALL_FEATURES)

	static_cols = ['area', 'total_occupants', 'num_rooms']
	aggregate_cols = [x for x in appliance_df.columns if "aggregate" in x]
	appliance_cols = [x for x in appliance_df.columns if appliance in x]
	X_cols = np.concatenate([aggregate_cols, static_cols])

	for features in ['energy', 'energy_static'][:]:
		pred[appliance][features] = {}
		if features == "energy":
			cols = aggregate_cols
		else:
			cols = X_cols
		appliance_df_X = appliance_df[cols]
		appliance_df_X = appliance_df_X.div(appliance_df_X.max())
		for neighbours in range(1, 10):
			print appliance, neighbours, features
			pred[appliance][features][neighbours] = {}
			loo = LeaveOneOut()
			for train_ix, test_ix in loo.split(appliance_df.index[:]):

				test_home = appliance_df.index.values[test_ix][0]
				pred[appliance][features][neighbours][test_home] = {}

				for month in range(start, stop):
					try:
						# Those homes which have non-NaN value for that appliance, month combination
						candidate_homes = appliance_df[appliance+"_"+str(month)].dropna().index.values

						# Adding test home for computing correlation
						if test_home not in candidate_homes:
							candidate_homes = np.append(candidate_homes, test_home)

						best_nghbrs = appliance_df_X.ix[candidate_homes].T.corr().ix[test_home].sort_values().dropna().drop(test_home).tail(
							neighbours).index.values
						pred[appliance][features][neighbours][test_home][month] = appliance_df[appliance_cols].ix[best_nghbrs][appliance+"_"+str(month)].mean()

					except:
						print test_home
#if ALL_HOMES:
#	pickle.dump(pred, open('predictions/knn.pkl','w'))
#else:
#	pickle.dump(pred, open('predictions/knn_all_homes.pkl', 'w'))

