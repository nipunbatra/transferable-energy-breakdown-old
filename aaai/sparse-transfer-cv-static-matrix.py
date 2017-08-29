import datetime
import pickle

from sklearn.model_selection import train_test_split, KFold

from aaai18.common import compute_rmse_fraction, contri
from create_matrix import *
from tensor_custom_core import *

appliance_index = {appliance: APPLIANCES_ORDER.index(appliance) for appliance in APPLIANCES_ORDER}




APPLIANCES = ['fridge', 'hvac', 'wm', 'mw', 'oven', 'dw']

year = 2014

import os


def learn_HAT_adagrad_static(case, E_np_masked, K, a, b, num_iter=2000, lr=0.1, dis=False,
                            A_known=None, T_known=None, random_seed=0, eps=1e-8, beta=1):

	def cost_static_fact(H, A, T, F, K, E_np_masked, case, beta=1):
		HAT = multiply_case(H, A, T, case)
		mask = ~np.isnan(E_np_masked)
		error_1 = (HAT - E_np_masked)[mask].flatten()

		mask = ~np.isnan(K)
		error_2 = (K - np.dot(H, F))[mask].flatten()

		return np.sqrt((error_1 ** 2).mean()) + beta * np.sqrt((error_2 ** 2).mean())

	np.random.seed(random_seed)
	cost = cost_static_fact

	mg = multigrad(cost, argnums=[0, 1, 2, 3])

	params = {}
	params['M'], params['N'], params['O'] = E_np_masked.shape
	params['a'] = a
	params['b'] = b
	H_dim_chars = list(cases[case]['HA'].split(",")[0].strip())
	H_dim = tuple(params[x] for x in H_dim_chars)
	A_dim_chars = list(cases[case]['HA'].split(",")[1].split("-")[0].strip())
	A_dim = tuple(params[x] for x in A_dim_chars)
	T_dim_chars = list(cases[case]['HAT'].split(",")[1].split("-")[0].strip())
	T_dim = tuple(params[x] for x in T_dim_chars)
	H = np.random.rand(*H_dim)

	A = np.random.rand(*A_dim)
	T = np.random.rand(*T_dim)
	# F is h X c
	F = np.random.rand(a, K.shape[1])

	sum_square_gradients_H = np.zeros_like(H)
	sum_square_gradients_A = np.zeros_like(A)
	sum_square_gradients_T = np.zeros_like(T)
	sum_square_gradients_F = np.zeros_like(F)

	Hs = [H.copy()]
	As = [A.copy()]
	Ts = [T.copy()]
	Fs = [F.copy()]
	costs = []

	HATs = [multiply_case(H, A, T, 2)]

	# GD procedure
	for i in range(num_iter):
		del_h, del_a, del_t, del_f = mg(H, A, T, F, K, E_np_masked, case, beta)
		sum_square_gradients_H += eps + np.square(del_h)
		sum_square_gradients_A += eps + np.square(del_a)
		sum_square_gradients_T += eps + np.square(del_t)
		sum_square_gradients_F += eps + np.square(del_f)

		lr_h = np.divide(lr, np.sqrt(sum_square_gradients_H))
		lr_a = np.divide(lr, np.sqrt(sum_square_gradients_A))
		lr_t = np.divide(lr, np.sqrt(sum_square_gradients_T))
		lr_f = np.divide(lr, np.sqrt(sum_square_gradients_F))

		H -= lr_h * del_h
		A -= lr_a * del_a
		T -= lr_t * del_t
		F -= lr_f * del_f

		# Projection to non-negative space
		H[H < 0] = 1e-8
		A[A < 0] = 1e-8
		T[T < 0] = 1e-8
		F[F < 0] = 1e-8

		As.append(A.copy())
		Ts.append(T.copy())
		Hs.append(H.copy())
		Fs.append(H.copy())
		costs.append(cost(H, A, T, F, K, E_np_masked, case, beta))
		HATs.append(multiply_case(H, A, T, 2))
		if i % 500 == 0:
			if dis:
				print(costs[-1])
	return H, A, T, F, Hs, As, Ts, HATs, costs


def un_normalize(x, maximum, minimum):
	return (maximum - minimum) * x + minimum


n_splits = 10
case = 2

source, target, random_seed, train_percentage = sys.argv[1:]
name = "{}-{}-{}-{}".format(source, target, random_seed, train_percentage)
directory = os.path.expanduser('~/aaai2017/transfer-static-matrix-{}_{}/'.format(source, target))
if not os.path.exists(directory):
	os.makedirs(directory)
filename = os.path.expanduser('~/aaai2017/transfer-static-matrix-{}_{}/'.format(source, target) + name + '.pkl')

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

algo = 'adagrad'


random_seed = int(random_seed)
train_percentage = float(train_percentage)


H_known_target = target_static
H_known_source = source_static

np.random.seed(random_seed)


kf = KFold(n_splits=n_splits)

pred = {}
for appliance in APPLIANCES_ORDER:
	pred[appliance] = []
print(random_seed, train_percentage)
sys.stdout.flush()
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
	best_beta = 0
	best_lr = 1
	least_error = 1e6

	overall_df_inner = target_df.loc[train_ix]

	best_params_global[outer_loop_iteration] = {}
	count = 0
	for num_iterations_cv in range(100, 1400, 600):
		for num_season_factors_cv in range(2, 5, 2):
			for num_home_factors_cv in range(3, 6, 2):
				#for lr_cv in [0.1, 1.]:
				for lr_cv in [1]:

					#for beta_cv in [1e-3,  1e-1, 0.]:
					for beta_cv in [1e-3, 1e-1, 0.]:
						count += 1
						print(count, num_iterations_cv, num_home_factors_cv, num_season_factors_cv, lr_cv,  beta_cv)

						sys.stdout.flush()
						pred_inner = {}
						for train_inner, test_inner in inner_kf.split(overall_df_inner):

							train_ix_inner = overall_df_inner.index[train_inner]
							test_ix_inner = overall_df_inner.index[test_inner]

							H_source, A_source, T_source, F_source, Hs_source, As_source, Ts_source, HATs_source, costs_source = learn_HAT_adagrad_static(case=case, E_np_masked=source_tensor, K=source_static,
							                                                                           a=num_home_factors_cv, b=num_season_factors_cv, num_iter=num_iterations_cv, lr=lr_cv, dis=False,
							                                                                           beta=beta_cv)
							train_test_ix_inner = np.concatenate([test_ix_inner, train_ix_inner])
							df_t_inner, dfc_t_inner = target_df.loc[train_test_ix_inner], target_dfc.loc[train_test_ix_inner]
							tensor_inner = get_tensor(df_t_inner)
							tensor_copy_inner = tensor_inner.copy()
							# First n
							tensor_copy_inner[:len(test_ix_inner), 1:, :] = np.NaN
							H, A, T, F, Hs, As, Ts, HATs, costs = learn_HAT_adagrad_static(
								case, tensor_copy_inner, H_known_target[np.concatenate([test_inner, train_inner])],
								num_home_factors_cv, num_season_factors_cv, num_iterations_cv, lr_cv, False,
								A_known=A_source,
								beta=beta_cv)
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
								print(e)
								print(appliance)
								sys.stdout.flush()
						print("Error weighted on: {}".format(appliance_to_weight))
						sys.stdout.flush()
						err_weight = {}
						for appliance in appliance_to_weight:
							err_weight[appliance] = err[appliance]*contri[target][appliance]
						mean_err = pd.Series(err_weight).sum()
						if mean_err < least_error:
							best_num_iterations = num_iterations_cv
							best_num_season_factors = num_season_factors_cv
							best_num_home_factors = num_home_factors_cv
							best_lr = lr_cv
							least_error = mean_err
						print(mean_err, least_error, num_iterations_cv, num_home_factors_cv, num_season_factors_cv)
						sys.stdout.flush()
	best_params_global[outer_loop_iteration] = {'Iterations':best_num_iterations,
	                                            "Appliance Train Error": err,
	                                            'Num season factors':best_num_season_factors,
	                                            'Num home factors': best_num_home_factors,
	                                            "Least Train Error":least_error,
	                                            'Learning rate':best_lr,
	                                            "Best beta":best_beta}

	print("******* BEST PARAMS *******")
	print(best_params_global[outer_loop_iteration])
	print("******* BEST PARAMS *******")
	sys.stdout.flush()
	# Now we will be using the best parameter set obtained to compute the predictions
	H_source, A_source, T_source, F_source, Hs_source, As_source, Ts_source, HATs_source, costs_source = learn_HAT_adagrad_static(
		case, source_tensor, source_static,
		best_num_home_factors, best_num_season_factors, best_num_iterations, best_lr, False,
		beta=best_beta)


	num_test = len(test_ix)
	train_test_ix = np.concatenate([test_ix, train_ix])
	df_t, dfc_t = target_df.loc[train_test_ix], target_dfc.loc[train_test_ix]
	tensor = get_tensor(df_t)
	tensor_copy = tensor.copy()
	# First n
	tensor_copy[:num_test, 1:, :] = np.NaN


	H, A, T, F, Hs, As, Ts, HATs, costs = learn_HAT_adagrad_static(
		case, tensor_copy, H_known_target[np.concatenate([test, train])],
		best_num_home_factors, best_num_season_factors, best_num_iterations, best_lr, False,
		A_known=A_source,
		beta=best_beta)


	HAT = multiply_case(H, A, T, case)
	for appliance in APPLIANCES_ORDER:
		pred[appliance].append(pd.DataFrame(HAT[:num_test, appliance_index[appliance], :], index=test_ix))

for appliance in APPLIANCES_ORDER:
	pred[appliance] = pd.DataFrame(pd.concat(pred[appliance]))

out = {'Predictions':pred, 'Learning Params':best_params_global}

with open(filename, 'wb') as f:
	pickle.dump(out, f, pickle.HIGHEST_PROTOCOL)
