import sys
import pickle


from sklearn.model_selection import KFold

from create_matrix import *
from sklearn.model_selection import train_test_split, KFold

from tensor_custom_core import *
from create_matrix import *
from tensor_custom_core import *
from degree_days import dds
from common import compute_rmse_fraction, contri

appliance_index = {appliance: APPLIANCES_ORDER.index(appliance) for appliance in APPLIANCES_ORDER}

APPLIANCES = ['fridge', 'hvac', 'wm', 'mw', 'oven', 'dw']

year = 2014

import os


def un_normalize(x, maximum, minimum):
	return (maximum - minimum) * x + minimum


n_splits = 10
case = 2

source, static_fac, lam, random_seed, train_percentage = sys.argv[1:]


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
	tensor = get_tensor(df)
	static_region = df[['area', 'total_occupants', 'num_rooms']].copy()
	static_region['area'] = static_region['area'].div(4000)
	static_region['total_occupants'] = static_region['total_occupants'].div(8)
	static_region['num_rooms'] = static_region['num_rooms'].div(8)
	static_region = static_region.values
	return df, dfc, tensor, static_region


source_df, source_dfc, source_tensor, source_static = create_region_df_dfc_static(source, year)

pred = {}
sd = {}
out = {}
n_splits = 10
case = 2
n_iter = 1200

cost = 'l21'
algo = 'adagrad'

lam = float(lam)

random_seed = int(random_seed)
train_percentage = float(train_percentage)

if static_fac == 'None':
	H_known_source = None
else:
	H_known_source = source_static
np.random.seed(random_seed)

kf = KFold(n_splits=n_splits)

pred = {}

for appliance in APPLIANCES_ORDER:
	pred[appliance] = []
print(lam, static_fac, random_seed, train_percentage)
best_params_global = {}

for outer_loop_iteration, (train_max, test) in enumerate(kf.split(source_df)):
	# Just a random thing
	np.random.seed(10 * random_seed + 7 * outer_loop_iteration)
	np.random.shuffle(train_max)
	print("-" * 80)
	print("Progress: {}".format(100.0 * outer_loop_iteration / n_splits))
	num_train = int((train_percentage * len(train_max) / 100) + 0.5)
	if train_percentage == 100:
		train = train_max
		train_ix = source_df.index[train]
		# print("Train set {}".format(train_ix.values))
		test_ix = source_df.index[test]
	else:
		train, _ = train_test_split(train_max, train_size=train_percentage / 100.0)
		train_ix = source_df.index[train]
		# print("Train set {}".format(train_ix.values))
		test_ix = source_df.index[test]

	print("-" * 80)

	print("Test set {}".format(test_ix.values))

	print("-" * 80)
	print("Current Error, Least Error, #Iterations")

	### Inner CV loop to find the optimum set of params. In this case: the number of iterations
	inner_kf = KFold(n_splits=2)
	best_num_iterations = 0
	best_num_season_factors = 0
	best_num_home_factors = 0
	best_appliance_wise_err = {appliance: 1e6 for appliance in APPLIANCES_ORDER[1:]}
	least_error = 1e6

	overall_df_inner = source_df.loc[train_ix]
	best_params_global[outer_loop_iteration] = {}
	for num_iterations_cv in range(100, 1400, 400):
		for num_season_factors_cv in range(2, 5):
			for num_home_factors_cv in range(3, 6):
				pred_inner = {}
				for train_inner, test_inner in inner_kf.split(overall_df_inner):
					train_ix_inner = overall_df_inner.index[train_inner]
					test_ix_inner = overall_df_inner.index[test_inner]

					train_test_ix_inner = np.concatenate([test_ix_inner, train_ix_inner])
					df_t_inner, dfc_t_inner = source_df.loc[train_test_ix_inner], source_dfc.loc[train_test_ix_inner]
					tensor_inner = get_tensor(df_t_inner)
					tensor_copy_inner = tensor_inner.copy()
					# First n
					tensor_copy_inner[:len(test_ix_inner), 1:, :] = np.NaN
					if static_fac != 'None':
						H, A, T, Hs, As, Ts, HATs, costs = learn_HAT_adagrad(case, tensor_copy_inner,
						                                                     num_home_factors_cv,
						                                                     num_season_factors_cv,
						                                                     num_iter=num_iterations_cv, lr=1,
						                                                     dis=False,
						                                                     cost_function=cost,
						                                                     H_known=H_known_source[
							                                                     np.concatenate(
								                                                     [test_inner, train_inner])],
						                                                     penalty_coeff=lam)
					else:
						H, A, T, Hs, As, Ts, HATs, costs = learn_HAT_adagrad(case, tensor_copy_inner,
						                                                     num_home_factors_cv,
						                                                     num_season_factors_cv,
						                                                     num_iter=num_iterations_cv, lr=1,
						                                                     dis=False,
						                                                     cost_function=cost,
						                                                     penalty_coeff=lam)
					HAT = multiply_case(H, A, T, case)
					for appliance in APPLIANCES_ORDER:
						if appliance not in pred_inner:
							pred_inner[appliance] = []

						pred_inner[appliance].append(
							pd.DataFrame(HAT[:len(test_ix_inner), appliance_index[appliance], :],
							             index=test_ix_inner))

				err = {}
				appliance_to_weight = []
				for appliance in APPLIANCES_ORDER[1:]:
					pred_inner[appliance] = pd.DataFrame(pd.concat(pred_inner[appliance]))

					try:
						if appliance == "hvac":
							err[appliance] = \
								compute_rmse_fraction(appliance, pred_inner[appliance][range(4, 10)], 'SanDiego')[2]
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
					err_weight[appliance] = err[appliance] * contri[source][appliance]
				mean_err = pd.Series(err_weight).sum()
				if mean_err < least_error:
					best_num_iterations = num_iterations_cv
					best_num_season_factors = num_season_factors_cv
					best_num_home_factors = num_home_factors_cv
					least_error = mean_err
					best_appliance_wise_err = err
				print(mean_err, least_error, num_iterations_cv, num_home_factors_cv, num_season_factors_cv)
	best_params_global[outer_loop_iteration] = {'Iterations': best_num_iterations,
	                                            "Appliance Train Error": best_appliance_wise_err,
	                                            'Num season factors': best_num_season_factors,
	                                            'Num home factors': best_num_home_factors,
	                                            "Least Train Error": least_error}

	print("******* BEST PARAMS *******")
	print(best_params_global[outer_loop_iteration])
	print("******* BEST PARAMS *******")

	num_test = len(test_ix)
	train_test_ix = np.concatenate([test_ix, train_ix])
	df_t, dfc_t = source_df.loc[train_test_ix], source_dfc.loc[train_test_ix]
	tensor = get_tensor(df_t)
	tensor_copy = tensor.copy()
	# First n
	tensor_copy[:num_test, 1:, :] = np.NaN
	if static_fac != 'None':
		H, A, T, Hs, As, Ts, HATs, costs = learn_HAT_adagrad(case, tensor_copy, best_num_home_factors,
		                                                     best_num_season_factors,
		                                                     num_iter=best_num_iterations, lr=1, dis=False,
		                                                     cost_function=cost,

		                                                     H_known=H_known_source[np.concatenate([test, train])],
		                                                     penalty_coeff=lam)
	else:
		H, A, T, Hs, As, Ts, HATs, costs = learn_HAT_adagrad(case, tensor_copy, best_num_home_factors,
		                                                     best_num_season_factors,
		                                                     num_iter=best_num_iterations, lr=1, dis=False,
		                                                     cost_function=cost,
		                                                     penalty_coeff=lam)

	HAT = multiply_case(H, A, T, case)
	for appliance in APPLIANCES_ORDER:
		pred[appliance].append(pd.DataFrame(HAT[:num_test, appliance_index[appliance], :], index=test_ix))

for appliance in APPLIANCES_ORDER:
	pred[appliance] = pd.DataFrame(pd.concat(pred[appliance]))

name = "{}-{}-{}-{}-{}-{}".format(source, static_fac, lam*1., random_seed, train_percentage*1., cost)
directory = os.path.expanduser('~/aaai2017/normal_{}_{}/'.format(source, cost))
if not os.path.exists(directory):
	os.makedirs(directory)
filename = os.path.expanduser('~/aaai2017/normal_{}_{}/'.format(source, cost) + name + '.pkl')

out = {'Predictions': pred, 'Learning Params': best_params_global}


with open(filename, 'wb') as f:
	pickle.dump(out, f, pickle.HIGHEST_PROTOCOL)
