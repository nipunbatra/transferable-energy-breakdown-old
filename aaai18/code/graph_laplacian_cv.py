"""
This module computes results for transfer learning

>>>python graph_laplacian.py setting case static_use source target random_seed train_percentage
setting: transfer or normal
case: 1, 2, 3, 4; 2 is for our proposed approach, 4 is for standard TF
static_use: "True" or "False"- If False, we don't use static household properties 
and the corresponding laplacian penalty term is set to 0
constant_use: "True" or "False" - If True, we add the constraint that one column of seasonal factors to be 1.
source:
target:
random_seed:
train_percentage:

NB: Prediction region is always called target. So, if we are doing n
normal learning on SD, we don't care about source, but target will be SD

Example:
# Transfer learning from Austin -> SD, case 2, 10% data used, 0th random seed, static_data used
>>> python graph_laplacian.py transfer 2 True Austin SanDiego 0 10

# Normal training in SD, case 2, 10% data used, 0th random seed, static data used
>>> python graph_laplacian.py normal 2 True None SanDiego 0 10

TODO: mention the objective being solved here

"""

import datetime
from sklearn.model_selection import train_test_split, KFold
from common import compute_rmse_fraction, contri, get_tensor, create_region_df_dfc_static
from create_matrix import *
from tensor_custom_core_all import *
import multiprocessing as mp
import pickle
global source, target
global case
global source_df, source_dfc, source_tensor, source_static
global target_df, target_dfc, target_tensor, target_static
global source_L, target_L
global T_constant

appliance_index = {appliance: APPLIANCES_ORDER.index(appliance) for appliance in APPLIANCES_ORDER}
APPLIANCES = ['fridge', 'hvac', 'wm', 'mw', 'oven', 'dw']
year = 2014

setting, case, constant_use, static_use, source, target, random_seed, train_percentage, start, stop = sys.argv[1:]
start = int(start)
stop = int(stop)
case = int(case)
train_percentage = float(train_percentage)
random_seed = int(random_seed)

if static_use == "True":
	# Use non-zero value of penalty
	lambda_cv_range = [0, 0.001, 0.01, 0.1]
else:
	lambda_cv_range = [0]

A_store = pickle.load(open(os.path.expanduser('~/git/scalable-nilm/aaai18/predictions/case-{}-graph_{}_{}_all_As.pkl'.format(case, source, constant_use)), 'r'))
source_df, source_dfc, source_tensor, source_static = create_region_df_dfc_static(source, year)
target_df, target_dfc, target_tensor, target_static = create_region_df_dfc_static(target, year)

# # using cosine similarity to compute L
source_L = get_L(source_static)
target_L = get_L(target_static)

if setting=="transfer":
	name = "{}-{}-{}-{}".format(source, target, random_seed, train_percentage)
else:
	name = "{}-{}-{}".format(target, random_seed, train_percentage)

# Seasonal constant constraints
if constant_use == 'True':
	T_constant = np.ones(stop-start).reshape(-1 , 1)
else:
	T_constant = None
# End

<<<<<<< HEAD
=======

>>>>>>> 4b8032832f00c39c105e8705285545de524db60a
n_splits = 10


best_params_global = {}
kf = KFold(n_splits=n_splits)


best_learning_rate = 0.1
best_num_iterations = 1300
best_num_season_factors = 2
best_num_home_factors = 3
best_lam = 0
least_error = 1e6

count = 0
error = []
params = {}
H_factors = {}
result = {}
result_app = {}
for learning_rate_cv in [0.1, 0.5, 1, 2]:
<<<<<<< HEAD
	H_factors[learning_rate_cv] = {}
	result[learning_rate_cv] = {}
	result_app[learning_rate_cv] = {}
	for num_iterations_cv in [1300, 700, 100][:]:
		H_factors[learning_rate_cv][num_iterations_cv] = {}
		result[learning_rate_cv][num_iterations_cv] = {}
		result_app[learning_rate_cv][num_iterations_cv] = {}
		for num_season_factors_cv in range(2, 5)[:]:
			H_factors[learning_rate_cv][num_iterations_cv][num_season_factors_cv] = {}
			result[learning_rate_cv][num_iterations_cv][num_season_factors_cv] = {}
			result_app[learning_rate_cv][num_iterations_cv][num_season_factors_cv] = {}
			for num_home_factors_cv in range(3, 6)[:]:
				H_factors[learning_rate_cv][num_iterations_cv][num_season_factors_cv][num_home_factors_cv] = {}
				result[learning_rate_cv][num_iterations_cv][num_season_factors_cv][num_home_factors_cv] = {}
				result_app[learning_rate_cv][num_iterations_cv][num_season_factors_cv][num_home_factors_cv] = {}
=======
	for num_iterations_cv in [1300, 700, 100][:]:
		for num_season_factors_cv in range(2, 5)[:]:
			for num_home_factors_cv in range(3, 6)[:]:
>>>>>>> 4b8032832f00c39c105e8705285545de524db60a
				if case == 4:
					if num_home_factors_cv!=num_season_factors_cv:
						print("Case 4 needs equal # dimensions. Skipping")
						sys.stdout.flush()

						continue
				for lam_cv in lambda_cv_range:
					if setting == 'transfer':
						A_source = A_store[learning_rate_cv][num_season_factors_cv][num_home_factors_cv][lam_cv][num_iterations_cv]
					else: 
						A_source = None

					
					# params[count] = []
					# params[count].extend((overall_df_inner, learning_rate_cv, num_iterations_cv, num_season_factors_cv, num_home_factors_cv, lam_cv, A_source))
					# count += 1
					pred = {}

					for appliance in APPLIANCES_ORDER:
						pred[appliance] = []
					for outer_loop_iteration, (train_max, test) in enumerate(kf.split(target_df)):
						# Just a random thing
						print num_iterations_cv, num_season_factors_cv, num_home_factors_cv, lam_cv
						np.random.seed(10 * random_seed + 7 * outer_loop_iteration)
						np.random.shuffle(train_max)
						print("-" * 80)
						print("Progress: {}".format(100.0 * outer_loop_iteration / n_splits))
						print(datetime.datetime.now())
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
							train, _ = train_test_split(train_max, train_size=train_percentage / 100.0)
							train_ix = target_df.index[train]
							test_ix = target_df.index[test]
						print train_ix

						print("-" * 80)
						print("Test set {}".format(test_ix.values))
						print("-" * 80)
						print("Current Error, Least Error, #Iterations")

						num_test = len(test_ix)
						train_test_ix = np.concatenate([test_ix, train_ix])
						df_t, dfc_t = target_df.loc[train_test_ix], target_dfc.loc[train_test_ix]
						tensor = get_tensor(df_t)
						tensor_copy = tensor.copy()
						# First n
						
						L = target_L[np.ix_(np.concatenate([test, train]), np.concatenate([test, train]))]

						H, A, T, Hs, As, Ts, HATs, costs = learn_HAT_adagrad_graph(case, tensor_copy, L,
																				  num_home_factors_cv,
																				  num_season_factors_cv,
																				  num_iter=num_iterations_cv, lr=learning_rate_cv, dis=False,
																				  lam=lam_cv, A_known=A_source, T_known=T_constant)

						HAT = multiply_case(H, A, T, case)
						for appliance in APPLIANCES_ORDER:
							pred[appliance].append(pd.DataFrame(HAT[:num_test, appliance_index[appliance], :], index=test_ix))

					# get the overall prediction error of all homes
					s = pd.concat(pred[appliance]).ix[target_df.index]
					err = {}
					for appliance in APPLIANCES_ORDER:
						if appliance=="hvac":
							err[appliance] = compute_rmse_fraction(appliance,s[range(5-start, 11-start)], target, start, stop)[2]
						else:   
							err[appliance] = compute_rmse_fraction(appliance, s,target, start, stop)[2]

					err_weight = {}
					for appliance in APPLIANCES_ORDER[1:]:
						err_weight[appliance] = err[appliance]*contri[target][appliance]
					mean_err = pd.Series(err_weight).sum()
					
					print(mean_err, least_error, learning_rate_cv, num_iterations_cv, num_home_factors_cv, num_season_factors_cv, lam_cv)
					if mean_err < least_error:
						best_learning_rate = learning_rate_cv
						best_num_iterations = num_iterations_cv
						best_num_season_factors = num_season_factors_cv
						best_num_home_factors = num_home_factors_cv
						best_lam = lam_cv
						least_error = mean_err


					
					# error.append(mean_err)
					# params[count] = []
					# params[count].extend((learning_rate_cv, num_iterations_cv, num_season_factors_cv, num_home_factors_cv, lam_cv))
						
					# count += 1

					# # get the error of each home
					# # 
					# s = pd.concat(pred[appliance]).ix[target_df.index]
					# err = {}

					# for appliance in APPLIANCES_ORDER:
					# 	if appliance=="hvac":
					# 		err[appliance] = compute_rmse_fraction(appliance,s[range(4, 10)], target)[3]
					# 	else:   
					# 		err[appliance] = compute_rmse_fraction(appliance, s,target)[3]

					# k = {}
					# for appliance in APPLIANCES_ORDER[1:]:
					    
					#     if appliance == 'hvac':
					#         start, end = 5, 11
					#     else:
					#         start, end = 1, 13

					#     error_home = pd.concat([err[appliance][appliance + "_{}".format(start)], 
					#                        err[appliance][appliance + "_{}".format(start+1)]],axis=1)
					    
<<<<<<< HEAD
					    for i in range(start+2, end):
					        error_home = pd.concat([error_home, err[appliance][appliance + "_{}".format(i)]], axis = 1)
					    app = np.sqrt((error_home**2).mean(axis=1))
					    k[appliance] = app
					result[learning_rate_cv][num_iterations_cv][num_season_factors_cv][num_home_factors_cv][lam_cv] = (pd.DataFrame(k).fillna(0)*pd.Series(contri[target])).sum(axis=1)
					result_app[learning_rate_cv][num_iterations_cv][num_season_factors_cv][num_home_factors_cv][lam_cv] = pd.DataFrame(k).fillna(0)


best_idx = np.argmin(error)
best_learning_rate, best_num_iterations, best_num_season_factors, best_num_home_factors, best_lam= params[best_idx]
least_error = error[best_idx]
print error
print params
print least_error
print params[best_idx]
=======
					#     for i in range(start+2, end):
					#         error_home = pd.concat([error_home, err[appliance][appliance + "_{}".format(i)]], axis = 1)
					#     app = np.sqrt((error_home**2).mean(axis=1))
					#     k[appliance] = app
					# result[learning_rate_cv][num_iterations_cv][num_season_factors_cv][num_home_factors_cv][lam_cv] = (pd.DataFrame(k).fillna(0)*pd.Series(contri[target])).sum(axis=1)


if setting == 'transfer':
	A_source = A_store[best_learning_rate][best_num_season_factors][best_num_home_factors][best_lam][best_num_iterations]
else: 
	A_source = None
pred = {}

for appliance in APPLIANCES_ORDER:
	pred[appliance] = []
for outer_loop_iteration, (train_max, test) in enumerate(kf.split(target_df)):
	# Just a random thing
	# print num_iterations_cv, num_season_factors_cv, num_home_factors_cv, lam_cv
	print best_learning_rate, best_num_iterations, best_num_season_factors, best_num_home_factors, best_lam
	np.random.seed(10 * random_seed + 7 * outer_loop_iteration)
	np.random.shuffle(train_max)
	print("-" * 80)
	print("Progress: {}".format(100.0 * outer_loop_iteration / n_splits))
	print(datetime.datetime.now())
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
		train, _ = train_test_split(train_max, train_size=train_percentage / 100.0)
		train_ix = target_df.index[train]
		test_ix = target_df.index[test]
	print train_ix

	print("-" * 80)
	print("Test set {}".format(test_ix.values))
	print("-" * 80)
	print("Current Error, Least Error, #Iterations")

	num_test = len(test_ix)
	train_test_ix = np.concatenate([test_ix, train_ix])
	df_t, dfc_t = target_df.loc[train_test_ix], target_dfc.loc[train_test_ix]
	tensor = get_tensor(df_t)
	tensor_copy = tensor.copy()
	# First n
	
	L = target_L[np.ix_(np.concatenate([test, train]), np.concatenate([test, train]))]

	H, A, T, Hs, As, Ts, HATs, costs = learn_HAT_adagrad_graph(case, tensor_copy, L,
															  best_num_home_factors,
															  best_num_season_factors,
															  num_iter=best_num_iterations, lr=best_learning_rate, dis=False,
															  lam=best_lam, A_known=A_source, T_known=T_constant)

	HAT = multiply_case(H, A, T, case)
	for appliance in APPLIANCES_ORDER:
		pred[appliance].append(pd.DataFrame(HAT[:num_test, appliance_index[appliance], :], index=test_ix))
>>>>>>> 4b8032832f00c39c105e8705285545de524db60a
# print result[0.1][1300][2][3][0]

for appliance in APPLIANCES_ORDER:
	pred[appliance] = pd.DataFrame(pd.concat(pred[appliance]))

if setting=="transfer":
	name = "{}-{}-{}-{}".format(source, target, random_seed, train_percentage)
else:
	name = "{}-{}-{}".format(target, random_seed, train_percentage)


directory = os.path.expanduser('~/git/scalable-nilm/aaai18/predictions/TF-all/{}/case-{}/{}/{}'.format(setting, case, static_use, constant_use))
if not os.path.exists(directory):
	os.makedirs(directory)
filename = os.path.join(directory, name + '.pkl')

if os.path.exists(filename):
	print("File already exists. Quitting.")

out = {'Predictions': pred, 'Learning Params': best_params_global}

with open(filename, 'wb') as f:
<<<<<<< HEAD
	pickle.dump(result, f, pickle.HIGHEST_PROTOCOL)

filename = os.path.join(directory, name + '_error_home_app.pkl')
with open(filename, 'wb') as f:
	pickle.dump(result_app, f, pickle.HIGHEST_PROTOCOL)
=======
	pickle.dump(out, f, pickle.HIGHEST_PROTOCOL)
>>>>>>> 4b8032832f00c39c105e8705285545de524db60a
