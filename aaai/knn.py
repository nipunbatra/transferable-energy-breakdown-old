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
for appliance in APPLIANCES:
	if appliance == "hvac":
		start, stop = 5, 11
	else:
		start, stop = 1, 13
	pred[appliance] = {}
	appliance_df = create_matrix_region_appliance_year(region, year, appliance, all_features=ALL_FEATURES)
	min_max_scaler = preprocessing.MinMaxScaler()
	static_cols = ['area', 'total_occupants', 'num_rooms']
	aggregate_cols = [x for x in appliance_df.columns if "aggregate" in x]
	X_cols = np.concatenate([aggregate_cols, static_cols])

	for features in ['energy', 'energy_static']:
		pred[appliance][features] = {}
		if features == "energy":
			cols = aggregate_cols
		else:
			cols = X_cols
		appliance_df_X = appliance_df[cols]
		appliance_df_X = min_max_scaler.fit_transform(appliance_df_X)
		for neighbours in range(1, 10):
			print neighbours, features, appliance
			pred[appliance][features][neighbours] = {}
			loo = LeaveOneOut()
			for train_ix, test_ix in loo.split(appliance_df):
				test_home = appliance_df.index.values[test_ix][0]
				pred[appliance][features][neighbours][test_home] = {}
				X_train = appliance_df_X[train_ix]
				X_test = appliance_df_X[test_ix]
				train_indices = appliance_df.index.values[train_ix]
				for month in range(start, stop):
					Y_train = appliance_df.ix[train_indices]['%s_%d' % (appliance, month)]
					Y_test = appliance_df.ix[test_home]['%s_%d' % (appliance, month)]
					knn = KNeighborsRegressor(n_neighbors=neighbours)
					knn.fit(X_train, Y_train)
					pred[appliance][features][neighbours][test_home][month] = knn.predict(X_test)[0]
			pred[appliance][features][neighbours] = pd.DataFrame(pred[appliance][features][neighbours]).T
if ALL_HOMES:
	pickle.dump(pred, open('predictions/knn.pkl','w'))
else:
	pickle.dump(pred, open('predictions/knn_all_homes.pkl', 'w'))

