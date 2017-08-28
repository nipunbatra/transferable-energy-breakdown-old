import sys
import pickle
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
import datetime

def un_normalize(x, maximum, minimum):
	return (maximum - minimum) * x + minimum


n_splits = 10
case = 2

source, target, static_fac, lam, random_seed, train_percentage, cost = sys.argv[1:]
name = "{}-{}-{}-{}-{}-{}-{}".format(source, target, static_fac, lam, random_seed, train_percentage, cost)
directory = os.path.expanduser('~/aaai2017/approach-2-pca-transfer_{}_{}_{}/'.format(source, target, cost))
if not os.path.exists(directory):
	os.makedirs(directory)
filename = os.path.expanduser('~/aaai2017/approach-2-pca-transfer_{}_{}_{}/'.format(source, target, cost) + name + '.pkl')

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
case = 2
n_iter = 1200

algo = 'adagrad'

lam = float(lam)

random_seed = int(random_seed)
train_percentage = float(train_percentage)

if static_fac =='None':
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

import autograd.numpy as np
from wpca import WPCA


def find_season_basis(tensor, n_components=2):
	data = tensor.reshape(tensor.shape[0] * 7, 12)
	weights = np.ones_like(data)
	weights[np.isnan(data)] = 0
	pca = WPCA(n_components=n_components).fit(data, weights=weights)
	return pca


def find_home_basis(tensor, n_components=2):
	data = np.sum(tensor, axis=2)

	weights = np.ones_like(data)
	weights[np.isnan(data)] = 0
	pca = WPCA(n_components=n_components).fit(data, weights=weights)
	return pca


def learning_pca(case, tensor, num_home_f, num_season_f, num_iter=2000, lr=0.1, dis=False, cost_function='abs',
                 random_seed=0, eps=1e-8, lam=0.0, A_known=None, eta=None):

	def cost_abs(H, A, T, eta, tensor, case, lam):
		HAT = multiply_case(H, A, T, case)
		mask = ~np.isnan(tensor)
		error = (HAT - tensor)[mask].flatten()
		error_2 = np.sum(eta - np.sum(A, axis=2))

		return np.sqrt((error ** 2).mean()) + lam * error_2

	T = find_season_basis(tensor, num_season_f).components_.T
	if eta is None:
		eta = find_home_basis(tensor, num_home_f).components_.T

	cost = cost_abs
	mg = multigrad(cost, argnums=[0, 1])

	params = {}
	params['M'], params['N'], params['O'] = tensor.shape
	params['a'] = num_home_f
	params['b'] = num_season_f
	H_dim_chars = list(cases[case]['HA'].split(",")[0].strip())
	H_dim = tuple(params[x] for x in H_dim_chars)
	A_dim_chars = list(cases[case]['HA'].split(",")[1].split("-")[0].strip())
	A_dim = tuple(params[x] for x in A_dim_chars)
	T_dim_chars = list(cases[case]['HAT'].split(",")[1].split("-")[0].strip())
	T_dim = tuple(params[x] for x in T_dim_chars)
	H = np.random.rand(*H_dim)

	A = np.random.rand(*A_dim)
	# return T, eta, H, A

	sum_square_gradients_H = np.zeros_like(H)
	sum_square_gradients_A = np.zeros_like(A)

	Hs = [H.copy()]
	As = [A.copy()]
	costs = [cost(H, A, T, eta, tensor, case, lam)]
	HATs = [multiply_case(H, A, T, 2)]

	# GD procedure
	for i in range(num_iter):
		del_h, del_a = mg(H, A, T, eta, tensor, case, lam)
		sum_square_gradients_H += eps + np.square(del_h)
		sum_square_gradients_A += eps + np.square(del_a)

		lr_h = np.divide(lr, np.sqrt(sum_square_gradients_H))
		lr_a = np.divide(lr, np.sqrt(sum_square_gradients_A))

		H -= lr_h * del_h
		A -= lr_a * del_a

		"""
		# Projection to non-negative space
		H[H < 0] = 1e-8
		A[A < 0] = 1e-8
		T[T < 0] = 1e-8
		"""

		As.append(A.copy())
		Hs.append(H.copy())
		costs.append(cost(H, A, T, eta, tensor, case, lam))
		HATs.append(multiply_case(H, A, T, 2))
		if i % 100 == 0:
			if dis:
				print(cost(H, A, T, eta, tensor, case, lam))
	return H, A, T, Hs, As, HATs, costs

def learning_pca_known_A(case, tensor, num_home_f, num_season_f, num_iter=2000, lr=0.1, dis=False, cost_function='abs',
                 random_seed=0, eps=1e-8, lam=0.0, A_known=None):
	def find_season_basis(tensor, n_components=2):
		data = tensor.reshape(tensor.shape[0] * 7, 12)
		weights = np.ones_like(data)
		weights[np.isnan(data)] = 0
		pca = WPCA(n_components=n_components).fit(data, weights=weights)
		return pca

	def find_home_basis(tensor, n_components=2):
		data = np.sum(tensor, axis=2)

		weights = np.ones_like(data)
		weights[np.isnan(data)] = 0
		pca = WPCA(n_components=n_components).fit(data, weights=weights)
		return pca

	def cost_abs(H, A, T, eta, tensor, case, lam):
		HAT = multiply_case(H, A, T, case)
		mask = ~np.isnan(tensor)
		error = (HAT - tensor)[mask].flatten()
		error_2 = np.sum(eta - np.sum(A, axis=2))

		return np.sqrt((error ** 2).mean()) + lam * error_2

	T = find_season_basis(tensor, num_season_f).components_.T
	eta = find_home_basis(tensor, num_home_f).components_.T

	cost = cost_abs
	from autograd import grad
	mg = grad(cost)

	params = {}
	params['M'], params['N'], params['O'] = tensor.shape
	params['a'] = num_home_f
	params['b'] = num_season_f
	H_dim_chars = list(cases[case]['HA'].split(",")[0].strip())
	H_dim = tuple(params[x] for x in H_dim_chars)

	H = np.random.rand(*H_dim)


	sum_square_gradients_H = np.zeros_like(H)
	A = A_known
	Hs = [H.copy()]
	costs = [cost(H, A, T, eta, tensor, case, lam)]
	HATs = [multiply_case(H, A, T, 2)]

	# GD procedure
	for i in range(num_iter):
		del_h = mg(H, A, T, eta, tensor, case, lam)
		sum_square_gradients_H += eps + np.square(del_h)

		lr_h = np.divide(lr, np.sqrt(sum_square_gradients_H))

		H -= lr_h * del_h

		"""
		# Projection to non-negative space
		H[H < 0] = 1e-8
		A[A < 0] = 1e-8
		T[T < 0] = 1e-8
		"""

		Hs.append(H.copy())
		costs.append(cost(H, A, T, eta, tensor, case, lam))
		HATs.append(multiply_case(H, A, T, 2))
		if i % 100 == 0:
			if dis:
				print(cost(H, A, T, eta, tensor, case, lam))
	return H, A, T, Hs, As, HATs, costs


import datetime

best_params_global = {}

for outer_loop_iteration, (train_max, test) in enumerate(kf.split(target_df)):
	# Just a random thing
	np.random.seed(10 * random_seed + 7*outer_loop_iteration)
	np.random.shuffle(train_max)
	print("-" * 80)
	print(datetime.datetime.now())
	print("Progress: {}".format(100.0*outer_loop_iteration/n_splits))
	sys.stdout.flush()
	num_train = int((train_percentage * len(train_max) / 100) + 0.5)
	if train_percentage == 100:
		train = train_max
		train_ix = target_df.index[train]
		# print("Train set {}".format(train_ix.values))
		test_ix = target_df.index[test]
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
	sys.stdout.flush()


	### Inner CV loop to find the optimum set of params. In this case: the number of iterations
	inner_kf = KFold(n_splits=2)

	best_num_iterations = 0
	best_num_season_factors = 0
	best_num_home_factors = 0
	least_error = 1e6
	best_lam = 1e-8

	overall_df_inner = target_df.loc[train_ix]

	best_params_global[outer_loop_iteration] = {}
	for num_iterations_cv in range(100, 1400, 600):
		for num_season_factors_cv in range(2, 5, 2):
			for num_home_factors_cv in range(3, 6, 2):
				for lam_cv in [1e-10, 1e-8, 1e-6]:
					pred_inner = {}
					for train_inner, test_inner in inner_kf.split(overall_df_inner):

						train_ix_inner = overall_df_inner.index[train_inner]
						test_ix_inner = overall_df_inner.index[test_inner]

						H_source, A_source, T_source, Hs, As, HATs, costs = learning_pca(case, source_tensor, num_home_factors_cv,
						                                            num_season_factors_cv, num_iter=num_iterations_cv,
						                                            lr=1,
						                                            dis=False, cost_function='abs', random_seed=0,
						                                            eps=1e-8, lam=lam_cv)
						train_test_ix_inner = np.concatenate([test_ix_inner, train_ix_inner])
						df_t_inner, dfc_t_inner = target_df.loc[train_test_ix_inner], target_dfc.loc[train_test_ix_inner]
						tensor_inner = get_tensor(df_t_inner)
						tensor_copy_inner = tensor_inner.copy()
						# First n
						tensor_copy_inner[:len(test_ix_inner), 1:, :] = np.NaN
						eta_source = find_home_basis(source_tensor, num_home_factors_cv).components_.T
						H, A, T, Hs, As, HATs, costs = learning_pca(case, tensor_copy_inner, num_home_factors_cv,
						                                            num_season_factors_cv, num_iter=num_iterations_cv,
						                                            lr=1,
						                                            dis=False, cost_function='abs', random_seed=0,
						                                            eps=1e-8, lam=lam_cv, eta=eta_source)

						HAT = multiply_case(H, A, T, case)
						for appliance in APPLIANCES_ORDER:
							if appliance not in pred_inner:
								pred_inner[appliance] = []

							pred_inner[appliance].append(pd.DataFrame(HAT[:len(test_ix_inner), appliance_index[appliance], :], index=test_ix_inner))

					err = {}
					appliance_to_weight = []
					for appliance in APPLIANCES_ORDER[1:]:
						pred_inner[appliance] = pd.DataFrame(pd.concat(pred_inner[appliance]))

						try:
							if appliance =="hvac":
								err[appliance] = compute_rmse_fraction(appliance, pred_inner[appliance][range(4, 10)], target)[2]
							else:
								err[appliance] = compute_rmse_fraction(appliance, pred_inner[appliance], target)[2]
							appliance_to_weight.append(appliance)
						except Exception, e:
							# This appliance does not have enough samples. Will not be
							# weighed
							pass

					err_weight = {}
					for appliance in appliance_to_weight:
						err_weight[appliance] = err[appliance]*contri[target][appliance]
					mean_err = pd.Series(err_weight).sum()
					if mean_err < least_error:
						best_num_iterations = num_iterations_cv
						best_num_season_factors = num_season_factors_cv
						best_num_home_factors = num_home_factors_cv
						least_error = mean_err
					print(mean_err, least_error, num_iterations_cv, num_home_factors_cv, num_season_factors_cv, lam_cv)
					sys.stdout.flush()
	best_params_global[outer_loop_iteration] = {'Iterations':best_num_iterations,
	                                            "Appliance Train Error": err,
	                                            'Num season factors':best_num_season_factors,
	                                            'Num home factors': best_num_home_factors,
	                                            "Least Train Error":least_error}

	print("******* BEST PARAMS *******")
	print(best_params_global[outer_loop_iteration])
	print("******* BEST PARAMS *******")
	print(sys.stdout.flush())
	# Now we will be using the best parameter set obtained to compute the predictions

	H_source, A_source, T_source, Hs, As, HATs, costs = learning_pca(case, source_tensor, best_num_home_factors,
	                                                                 best_num_season_factors,
	                                                                 num_iter=best_num_iterations,
	                                                                 lr=1,
	                                                                 dis=False, cost_function='abs', random_seed=0,
	                                                                 eps=1e-8, lam=best_lam)

	num_test = len(test_ix)
	train_test_ix = np.concatenate([test_ix, train_ix])
	df_t, dfc_t = target_df.loc[train_test_ix], target_dfc.loc[train_test_ix]
	tensor = get_tensor(df_t)
	tensor_copy = tensor.copy()
	# First n
	tensor_copy[:num_test, 1:, :] = np.NaN
	eta_source =  find_home_basis(source_tensor, best_num_home_factors).components_.T
	H, A, T, Hs, As, HATs, costs = learning_pca(case, tensor_copy_inner, num_home_factors_cv,
	                                            num_season_factors_cv, num_iter=num_iterations_cv,
	                                            lr=1,
	                                            dis=False, cost_function='abs', random_seed=0,
	                                            eps=1e-8, lam=lam_cv, eta=eta_source)

	HAT = multiply_case(H, A, T, case)
	for appliance in APPLIANCES_ORDER:
		pred[appliance].append(pd.DataFrame(HAT[:num_test, appliance_index[appliance], :], index=test_ix))

for appliance in APPLIANCES_ORDER:
	pred[appliance] = pd.DataFrame(pd.concat(pred[appliance]))

out = {'Predictions':pred, 'Learning Params':best_params_global}

with open(filename, 'wb') as f:
	pickle.dump(out, f, pickle.HIGHEST_PROTOCOL)
