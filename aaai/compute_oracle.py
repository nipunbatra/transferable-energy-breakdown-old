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

pred = {}
for appliance in APPLIANCES[:]:
	pred[appliance] = {}
	if appliance == "hvac":
		start, stop = 5, 11
	else:
		start, stop = 1, 13
	appliance_cols = ['%s_%d' %(appliance, month) for month in range(start, stop)]
	pred[appliance] = {}
	appliance_df = create_matrix_region_appliance_year(region, year, appliance)

	for test_home in appliance_df.index[:]:
		print test_home, appliance
		df = appliance_df[appliance_df.ix[test_home].dropna().index][appliance_cols]

		if len(df) > 2:
			# Closest index
			if len((df - df.ix[test_home]).drop(test_home).dropna()):
				# Only if there exist atleast one home having all the features
				best_nghbr = \
				(df - df.ix[test_home]).drop(test_home).dropna().apply(np.square).sum(axis=1).sort_values().index[0]
				best_prediction = df.ix[best_nghbr]
				pred[appliance][test_home] = best_prediction
	pred[appliance] = pd.DataFrame(pred[appliance]).T

pickle.dump(pred, open('predictions/knn_oracle.pkl','w'))


