
import sys

from sklearn.model_selection import KFold

from create_matrix import *
from sklearn.model_selection import train_test_split, KFold
from common import compute_rmse_fraction, contri
from tensor_custom_core import *
from create_matrix import *
from tensor_custom_core import *
from degree_days import dds

appliance_index = {appliance: APPLIANCES_ORDER.index(appliance) for appliance in APPLIANCES_ORDER}



APPLIANCES = ['fridge', 'hvac', 'wm', 'mw', 'oven', 'dw']

year = 2014

import os


def un_normalize(x, maximum, minimum):
	return (maximum - minimum) * x + minimum


n_splits = 10
case = 2

source, target, static_fac, lam, num_home_factors, num_season_factors, random_seed, train_percentage, cost = sys.argv[1:]
name = "{}-{}-{}-{}-{}-{}-{}-{}-{}".format(source, target, static_fac, lam, num_home_factors, num_season_factors, random_seed, train_percentage, cost)
directory = os.path.expanduser('~/aaai2017/transfer_{}_{}_{}/'.format(source, target, cost))
if not os.path.exists(directory):
	os.makedirs(directory)
filename = os.path.expanduser('~/aaai2017/transfer_{}_{}_{}/'.format(source, target, cost) + name + '.pkl')

if os.path.exists(filename):
	print("File already exists. Quitting.")
	#sys.exit(0)


def get_tensor(df):
	start, stop = 1, 13
	energy_cols = np.array(
		[['%s_%d' % (appliance, month) for month in range(start, stop)] for appliance in APPLIANCES_ORDER]).flatten()

	dfc = df.copy()

	df = dfc[energy_cols]

	tensor = df.values.reshape((len(df), 7, stop - start))
	return tensor


def create_region_df_dfc_static(region, year):
	df, dfc = create_matrix_single_region(region, year)
	if target=="Austin":
		df = df.head(50)
		dfc = df.head(50)
	tensor = get_tensor(df)
	static_region = df[['area', 'total_occupants', 'num_rooms']].copy()
	static_region['area'] = static_region['area'].div(4000)
	static_region['total_occupants'] = static_region['total_occupants'].div(8)
	static_region['num_rooms'] = static_region['num_rooms'].div(8)
	static_region =static_region.values
	return df, dfc, tensor, static_region

source_df, source_dfc, source_tensor, source_static = create_region_df_dfc_static(source, year)
target_df, target_dfc, target_tensor, target_static = create_region_df_dfc_static(target, year)

pred = {}
sd = {}
out = {}
n_splits = 10
NUM_RANDOM = 3
TRAIN_SPLITS = range(10, 110, 40)
case = 2
n_iter = 1200

algo = 'adagrad'

lam = float(lam)
num_home_factors = int(num_home_factors)
num_season_factors = int(num_season_factors)
random_seed = int(random_seed)
train_percentage = float(train_percentage)

if static_fac is 'None':
	H_known_target = None
	H_known_source = None
else:
	H_known_target = target_static
	H_known_source = source_static
np.random.seed(random_seed)


kf = KFold(n_splits=n_splits)

pred = {}
for appliance in APPLIANCES_ORDER:
	pred[appliance] = []
print(lam, static_fac, num_season_factors, num_home_factors, random_seed, train_percentage)
for train_max, test in kf.split(target_df):

	num_train = int((train_percentage * len(train_max) / 100) + 0.5)
	if train_percentage == 100:
		train = train_max
	else:
		train, _ = train_test_split(train_max, train_size=train_percentage / 100.0)
	train_ix = target_df.index[train]
	test_ix = target_df.index[test]

	print("-" * 80)
	print("Test set {}".format(test_ix.values))
	print("Train max set {}".format(target_df.index[train_max].values))
	print("Train set {}".format(train_ix.values))
	print("-"*80)
	print("Current Error, Least Error, #Iterations")


	### Inner CV loop to find the optimum set of params. In this case: the number of iterations
	inner_kf = KFold(n_splits=2)

	best_num_iterations = 0
	least_error = 1e6

	overall_df_inner = target_df.loc[train_ix]
	for num_iterations_cv in range(100, 600, 400):
		pred_inner = {}
		for train_inner, test_inner in inner_kf.split(overall_df_inner):

			train_ix_inner = overall_df_inner.index[train_inner]
			test_ix_inner = overall_df_inner.index[test_inner]
			#print(test_ix_inner)

			if static_fac == 'static':
				H_source, A_source, T_source, Hs, As, Ts, HATs, costs = learn_HAT_adagrad(case, source_tensor, num_home_factors,
				                                                                          num_season_factors,
				                                                                          num_iter=num_iterations_cv, lr=1, dis=False,
				                                                                          cost_function=cost,
				                                                                          H_known=H_known_source,
				                                                                          penalty_coeff=lam)
			else:
				H_source, A_source, T_source, Hs, As, Ts, HATs, costs = learn_HAT_adagrad(case, source_tensor, num_home_factors,
			                                                                          num_season_factors,
			                                                                          num_iter=num_iterations_cv, lr=1, dis=False,
			                                                                          cost_function=cost,
			                                                                          penalty_coeff=lam)
			train_test_ix_inner = np.concatenate([test_ix_inner, train_ix])
			df_t_inner, dfc_t_inner = target_df.loc[train_test_ix_inner], target_dfc.loc[train_test_ix_inner]
			tensor_inner = get_tensor(df_t_inner)
			tensor_copy_inner = tensor_inner.copy()
			# First n
			tensor_copy_inner[:len(test_ix_inner), 1:, :] = np.NaN
			if static_fac != 'None':
				H, A, T, Hs, As, Ts, HATs, costs = learn_HAT_adagrad(case, tensor_copy_inner, num_home_factors,
				                                                     num_season_factors,
				                                                     num_iter=num_iterations_cv, lr=1, dis=False,
				                                                     cost_function=cost,
				                                                     A_known=A_source,
				                                                     H_known=H_known_target[np.concatenate([test_inner, train])],
				                                                     penalty_coeff=lam)
			else:
				H, A, T, Hs, As, Ts, HATs, costs = learn_HAT_adagrad(case, tensor_copy_inner, num_home_factors,
				                                                     num_season_factors,
				                                                     num_iter=num_iterations_cv, lr=1, dis=False,
				                                                     cost_function=cost,
				                                                     A_known=A_source, penalty_coeff=lam)

			HAT = multiply_case(H, A, T, case)
			for appliance in APPLIANCES_ORDER:
				if appliance not in pred_inner:
					pred_inner[appliance] = []

				pred_inner[appliance].append(pd.DataFrame(HAT[:len(test_ix_inner), appliance_index[appliance], :], index=test_ix_inner))

		err = {}
		for appliance in APPLIANCES_ORDER[1:]:
			pred_inner[appliance] = pd.DataFrame(pd.concat(pred_inner[appliance]))
			if appliance =="hvac":
				err[appliance] = compute_rmse_fraction(appliance, pred_inner[appliance][range(4, 10)], 'SanDiego')[2]
			else:
				err[appliance] = compute_rmse_fraction(appliance, pred_inner[appliance], 'SanDiego')[2]
			err[appliance] = err[appliance]*contri[target][appliance]
		mean_err = pd.Series(err).sum()
		if mean_err < least_error:
			best_num_iterations = num_iterations_cv
			least_error = mean_err
		print(mean_err, least_error, num_iterations_cv)


	# Now we will be using the best parameter set obtained to compute the predictions

	if static_fac == 'static':
		H_source, A_source, T_source, Hs, As, Ts, HATs, costs = learn_HAT_adagrad(case, source_tensor, num_home_factors,
		                                                                          num_season_factors,
		                                                                          num_iter=best_num_iterations, lr=1,
		                                                                          dis=False,
		                                                                          cost_function=cost,
		                                                                          H_known=H_known_source,
		                                                                          penalty_coeff=lam)
	else:
		H_source, A_source, T_source, Hs, As, Ts, HATs, costs = learn_HAT_adagrad(case, source_tensor, num_home_factors,
		                                                                          num_season_factors,
		                                                                          num_iter=best_num_iterations, lr=1,
		                                                                          dis=False,
		                                                                          cost_function=cost,
		                                                                          penalty_coeff=lam)

	num_test = len(test_ix)
	train_test_ix = np.concatenate([test_ix, train_ix])
	df_t, dfc_t = target_df.loc[train_test_ix], target_dfc.loc[train_test_ix]
	tensor = get_tensor(df_t)
	tensor_copy = tensor.copy()
	# First n
	tensor_copy[:num_test, 1:, :] = np.NaN
	if static_fac is not None:
		H, A, T, Hs, As, Ts, HATs, costs = learn_HAT_adagrad(case, tensor_copy, num_home_factors, num_season_factors,
		                                                     num_iter=best_num_iterations, lr=1, dis=False, cost_function=cost,
		                                                     A_known=A_source,
		                                                     H_known=H_known_target[np.concatenate([test, train])],
		                                                     penalty_coeff=lam)
	else:
		H, A, T, Hs, As, Ts, HATs, costs = learn_HAT_adagrad(case, tensor_copy, num_home_factors, num_season_factors,
		                                                     num_iter=best_num_iterations, lr=1, dis=False, cost_function=cost,
		                                                     A_known=A_source, penalty_coeff=lam)

	HAT = multiply_case(H, A, T, case)
	for appliance in APPLIANCES_ORDER:
		pred[appliance].append(pd.DataFrame(HAT[:num_test, appliance_index[appliance], :], index=test_ix))

for appliance in APPLIANCES_ORDER:
	pred[appliance] = pd.DataFrame(pd.concat(pred[appliance]))

import pickle

with open(filename, 'wb') as f:
	pickle.dump(pred, f, pickle.HIGHEST_PROTOCOL)
