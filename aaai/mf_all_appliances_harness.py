from mf_core import *

import numpy as np
import pandas as pd
import sys
import os
from create_matrix import *
from sklearn.model_selection import LeaveOneOut
from sklearn.neighbors import KNeighborsRegressor
import pickle


from degree_days import dds
appliance_index = {appliance: APPLIANCES_ORDER.index(appliance) for appliance in APPLIANCES_ORDER}

APPLIANCES = ['fridge', 'hvac', 'wm', 'mw', 'oven', 'dw']

import sys

features, cost, all_features, latent_factors = sys.argv[1:]
latent_factors = int(latent_factors)
if all_features == 'True':
	all_features = True
else:
	all_features = False


region, year = 'SanDiego', 2014
df, dfc = create_matrix_single_region(region, year)
start, stop = 1, 13
energy_cols = np.array(
	[['%s_%d' % (appliance, month) for month in range(start, stop)] for appliance in APPLIANCES_ORDER]).flatten()

static_cols = ['area', 'total_occupants', 'num_rooms']
static_df = df[static_cols]
static_df = static_df.div(static_df.max())
weather_values = np.array(dds[2014][region][start - 1:stop - 1]).reshape(-1, 1)

dfc = df.copy()

df = dfc[energy_cols]
col_max = df.max().max()
col_min = df.min().min()

X_matrix, X_normalised, matrix_max, matrix_min = preprocess_all_appliances(df, dfc)
static_features = get_static_features(dfc, X_normalised)
if features == "energy":
	feature_comb = ['None']
else:
	feature_comb = ['occ', 'area', 'rooms']
idx_user, data_user = prepare_known_features(feature_comb, static_features, X_normalised)

loo = LeaveOneOut()
pred = {a:{} for a in APPLIANCES_ORDER}
for train_ix, test_ix in loo.split(df[:]):
	try:

		test_home = df.index.values[test_ix][0]
		print test_home
		A = create_matrix_factorised_all_appliances([test_home], X_normalised)
		X, Y, res = nmf_features(A=A, k=latent_factors, constant=0.01, regularisation=False,
		                         idx_user=idx_user, data_user=data_user,
		                         idx_item=None, data_item=None, MAX_ITERS=8, cost=cost)
		pred_df = {}
		for appliance in APPLIANCES_ORDER:
			appliance_cols = ['%s_%d' %(appliance, month) for month in range(1, 13)]

			pred[appliance][test_home] = create_prediction(test_home, X, Y, X_normalised, appliance,
		                            col_max, col_min, appliance_cols)
	except:
		pass


if region=="Austin":
	base_path = os.path.expanduser("~/scalable/mf_all_appliances/")
else:
	base_path = os.path.expanduser("~/scalable/sd/mf_all_appliances/")
if not os.path.exists(base_path):
	os.makedirs(base_path)
for appliance in APPLIANCES_ORDER:
	out = pd.DataFrame(pred[appliance]).T
	store_path = "/".join([str(x) for x in [appliance, features, cost, all_features, latent_factors]])
	store_path = os.path.join(base_path, store_path)
	if not os.path.exists(store_path):
		os.makedirs(store_path)
	store_path = store_path + "/data.csv"

	out.to_csv(store_path)

