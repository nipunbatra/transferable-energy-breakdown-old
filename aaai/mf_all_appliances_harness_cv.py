from mf_core import *

import numpy as np
import pandas as pd
import sys
import os
from create_matrix import *
from sklearn.model_selection import LeaveOneOut, KFold, train_test_split
from sklearn.neighbors import KNeighborsRegressor
import pickle
from common import compute_rmse_fraction, contri


from degree_days import dds
appliance_index = {appliance: APPLIANCES_ORDER.index(appliance) for appliance in APPLIANCES_ORDER}

APPLIANCES = ['fridge', 'hvac', 'wm', 'mw', 'oven', 'dw']

import sys

n_splits = 10

features, cost, all_features, latent_factors, random_seed, train_percentage = sys.argv[1:]
latent_factors = int(latent_factors)
random_seed = int(random_seed)
train_percentage = float(train_percentage)
if all_features == 'True':
	all_features = True
else:
	all_features = False


region, year = 'SanDiego', 2014
target = region
target_df, target_dfc = create_matrix_single_region(region, year)
start, stop = 1, 13
energy_cols = np.array(
	[['%s_%d' % (appliance, month) for month in range(start, stop)] for appliance in APPLIANCES_ORDER]).flatten()

static_cols = ['area', 'total_occupants', 'num_rooms']
static_df = target_df[static_cols]
static_df = static_df.div(static_df.max())
weather_values = np.array(dds[2014][region][start - 1:stop - 1]).reshape(-1, 1)

target_dfc = target_df.copy()

target_df = target_dfc[energy_cols]
col_max = target_df.max().max()
col_min = target_df.min().min()

X_matrix, X_normalised, matrix_max, matrix_min = preprocess_all_appliances(target_df, target_dfc)
static_features = get_static_features(target_dfc, X_normalised)
if features == "energy":
	feature_comb = ['None']
else:
	feature_comb = ['occ', 'area', 'rooms']
idx_user, data_user = prepare_known_features(feature_comb, static_features, X_normalised)

kf = KFold(n_splits=n_splits)

pred = {a:{} for a in APPLIANCES_ORDER}

best_params_global = {}
pred_df = {}
for outer_loop_iteration, (train_max, test) in enumerate(kf.split(target_df)):
	# Just a random thing
	np.random.seed(10 * random_seed + 7*outer_loop_iteration)
	np.random.shuffle(train_max)
	print("-" * 80)
	print("Progress: {}".format(100.0*outer_loop_iteration/n_splits))
	num_train = int((train_percentage * len(train_max) / 100) + 0.5)
	if train_percentage == 100:
		train = train_max
	else:
		# Sample `train_percentage` homes
		# An important condition here is that all homes should have energy data
		# for all appliances for atleast one month.
		SAMPLE_CRITERION_MET = False


		train, _ = train_test_split(train_max, train_size=train_percentage / 100.0)
		train_ix = target_df.index[train]
		#print("Train set {}".format(train_ix.values))
		test_ix = target_df.index[test]
		a = target_df.loc[train_ix]
		count_condition_violation = 0

	print("-" * 80)

	print("Test set {}".format(test_ix.values))


	print("-"*80)
	print("Current Error, Least Error, #Iterations")


	### Inner CV loop to find the optimum set of params. In this case: the number of iterations
	inner_kf = KFold(n_splits=2)

	best_num_iterations = 0
	best_num_season_factors = 0
	best_num_home_factors = 0
	least_error = 1e6

	overall_df_inner = target_df.loc[train_ix]

	best_params_global[outer_loop_iteration] = {}
	for num_iterations_cv in range(5, 10):
		for num_latent_factors_cv in range(2, 6):
			pred_inner = {}
			for train_inner, test_inner in inner_kf.split(overall_df_inner):

				train_ix_inner = overall_df_inner.index[train_inner]
				test_ix_inner = overall_df_inner.index[test_inner]
				train_test_ix_inner = np.concatenate([test_ix_inner, train_ix_inner])
				df_t_inner, dfc_t_inner = target_df.loc[train_test_ix_inner], target_dfc.loc[train_test_ix_inner]
				X_matrix, X_normalised, matrix_max, matrix_min = preprocess_all_appliances(df_t_inner, dfc_t_inner)

				static_features = get_static_features(dfc_t_inner, X_normalised)
				if features == "energy":
					feature_comb = ['None']
				else:
					feature_comb = ['occ', 'area', 'rooms']
				idx_user, data_user = prepare_known_features(feature_comb, static_features, X_normalised)

				A = create_matrix_factorised_all_appliances(test_ix_inner, X_normalised)
				X, Y, res = nmf_features(A=A, k=num_latent_factors_cv, constant=0.01, regularisation=False,
				                         idx_user=idx_user, data_user=data_user,
				                         idx_item=None, data_item=None, MAX_ITERS=8, cost=cost)

				pred_inner = {}
				for appliance in APPLIANCES_ORDER:
					if appliance not in pred_inner:
						pred_inner[appliance] = []
					appliance_cols = ['%s_%d' % (appliance, month) for month in range(1, 13)]

					pred_inner[appliance].append(create_prediction(test_ix_inner, X, Y, X_normalised, appliance,
					                                               col_max, col_min, appliance_cols))



			err = {}
			appliance_to_weight = []
			for appliance in APPLIANCES_ORDER[1:]:
				pred_inner[appliance] = pd.DataFrame(pd.concat(pred_inner[appliance]))

				try:
					if appliance =="hvac":
						err[appliance] = compute_rmse_fraction(appliance, pred_inner[appliance][['hvac_{}'.format(month) for month in range(4, 10)]], 'SanDiego')[2]
					else:
						err[appliance] = compute_rmse_fraction(appliance, pred_inner[appliance], 'SanDiego')[2]
					appliance_to_weight.append(appliance)
				except Exception, e:
					# This appliance does not have enough samples. Will not be
					# weighed
					print(e)
					print(appliance)
			print("Error weighted on: {}".format(appliance_to_weight))
			err_weight = {}
			for appliance in appliance_to_weight:
				err_weight[appliance] = err[appliance]*contri[target][appliance]
			mean_err = pd.Series(err_weight).sum()
			if mean_err < least_error:
				best_num_iterations = num_iterations_cv
				best_num_latent_factors = num_latent_factors_cv
				least_error = mean_err
			print(mean_err, least_error, num_iterations_cv, num_latent_factors_cv)
	best_params_global[outer_loop_iteration] = {'Iterations':best_num_iterations,
	                                            "Appliance Train Error": err,
	                                            'Num latent factors': best_num_latent_factors,
	                                            "Least Train Error":least_error}

	print("******* BEST PARAMS *******")
	print(best_params_global[outer_loop_iteration])
	print("******* BEST PARAMS *******")
	# Now we will be using the best parameter set obtained to compute the predictions



	num_test = len(test_ix)
	train_test_ix = np.concatenate([test_ix, train_ix])
	df_t, dfc_t = target_df.loc[train_test_ix], target_dfc.loc[train_test_ix]
	X_matrix, X_normalised, matrix_max, matrix_min = preprocess_all_appliances(df_t, dfc_t)

	static_features = get_static_features(dfc_t, X_normalised)
	if features == "energy":
		feature_comb = ['None']
	else:
		feature_comb = ['occ', 'area', 'rooms']
	idx_user, data_user = prepare_known_features(feature_comb, static_features, X_normalised)

	A = create_matrix_factorised_all_appliances(test_ix, X_normalised)
	X, Y, res = nmf_features(A=A, k=best_num_latent_factors, constant=0.01, regularisation=False,
	                         idx_user=idx_user, data_user=data_user,
	                         idx_item=None, data_item=None, MAX_ITERS=best_num_iterations, cost=cost)
	for appliance in APPLIANCES_ORDER[1:]:
		if appliance not in pred_df:
			pred_df[appliance] = []
		appliance_cols = ['%s_%d' % (appliance, month) for month in range(1, 13)]

		pred_df[appliance].append(create_prediction(test_ix, X, Y, X_normalised, appliance,
		                                               col_max, col_min, appliance_cols))


