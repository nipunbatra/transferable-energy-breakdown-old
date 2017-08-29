import pandas as pd
from create_matrix import *
from tensor_custom_core import *
from create_matrix import *
from tensor_custom_core import *
from degree_days import dds
import os
from degree_days import dds
import autograd.numpy as np
from sklearn.model_selection import train_test_split, KFold
from common import compute_rmse_fraction, contri
from sklearn.neighbors import NearestNeighbors
import pickle
import multiprocessing as mp
from scipy.spatial.distance import pdist, squareform

appliance_index = {appliance: APPLIANCES_ORDER.index(appliance) for appliance in APPLIANCES_ORDER}
APPLIANCES = ['fridge', 'hvac', 'wm', 'mw', 'oven', 'dw']
year = 2014


def un_normalize(x, maximum, minimum):
	return (maximum - minimum) * x + minimum

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

def distance(x, y):
	return np.linalg.norm(x - y)

def get_L_NN(X):
	nbrs = NearestNeighbors(n_neighbors=5, algorithm='ball_tree').fit(X)
	distances, indices = nbrs.kneighbors(X)
	n_sample, n_feature = X.shape
	W = np.zeros((n_sample, n_sample))
	for i in range(n_sample):
		for j in indices[i]:
			W[i][j] = 1
			W[j][i] = 1
	K = np.dot(W, np.ones((n_sample, n_sample)))
	D = np.diag(np.diag(K))
	return D - W

def get_L(X):
	W = 1-squareform(pdist(X, 'cosine'))
	W = np.nan_to_num(W)
	n_sample, n_feature = W.shape
	
	K = np.dot(W, np.ones((n_sample, n_sample)))
	D = np.diag(np.diag(K))
	return D - W

def cost_graph_laplacian(H, A, T, L, E_np_masked, lam, case):
	HAT = multiply_case(H, A, T, case)
	mask = ~np.isnan(E_np_masked)
	error_1 = (HAT - E_np_masked)[mask].flatten()
	
	HTL = np.dot(H.T, L)
	HTLH = np.dot(HTL, H)
	error_2 = np.trace(HTLH)
	
	return np.sqrt((error_1**2).mean()) + lam * error_2

def learn_HAT_adagrad_graph(case, E_np_masked, L, a, b, num_iter=2000, lr=0.01, dis=False, lam = 1, H_known=None,A_known=None, T_known=None, random_seed=0, eps=1e-8, penalty_coeff=0.0):

	np.random.seed(random_seed)
	cost = cost_graph_laplacian
	mg = multigrad(cost, argnums=[0, 1, 2])

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

	sum_square_gradients_H = np.zeros_like(H)
	sum_square_gradients_A = np.zeros_like(A)
	sum_square_gradients_T = np.zeros_like(T)

	Hs = [H.copy()]
	As = [A.copy()]
	Ts = [T.copy()]
	
	costs = [cost(H, A, T, L, E_np_masked, lam, case)]

	HATs = [multiply_case(H, A, T, case)]

	# GD procedure
	for i in range(num_iter):
		del_h, del_a, del_t = mg(H, A, T, L, E_np_masked, lam, case)
		sum_square_gradients_H += eps + np.square(del_h)
		sum_square_gradients_A += eps + np.square(del_a)
		sum_square_gradients_T += eps + np.square(del_t)

		lr_h = np.divide(lr, np.sqrt(sum_square_gradients_H))
		lr_a = np.divide(lr, np.sqrt(sum_square_gradients_A))
		lr_t = np.divide(lr, np.sqrt(sum_square_gradients_T))

		H -= lr_h * del_h
		A -= lr_a * del_a
		T -= lr_t * del_t
		# Projection to known values
		if H_known is not None:
			H = set_known(H, H_known)
		if A_known is not None:
			A = set_known(A, A_known)
		if T_known is not None:
			T = set_known(T, T_known)
		# Projection to non-negative space
		H[H < 0] = 1e-8
		A[A < 0] = 1e-8
		T[T < 0] = 1e-8

		As.append(A.copy())
		Ts.append(T.copy())
		Hs.append(H.copy())
		
		costs.append(cost(H, A, T, L, E_np_masked, lam, case))
		

		HATs.append(multiply_case(H, A, T, case))
		if i % 500 == 0:
			if dis:
				print(cost(H, A, T, L, E_np_masked, lam, case))
	return H, A, T, Hs, As, Ts, HATs, costs


global case
global source, source_df, source_dfc, source_tensor, source_static
global source_L
case = 2


source, random_seed, train_percentage = sys.argv[1:]
train_percentage = float(train_percentage)
random_seed = int(random_seed)

source_df, source_dfc, source_tensor, source_static = create_region_df_dfc_static(source, year)


# # using cosine similarity to compute L
source_L = get_L(source_static)



def compute_inner_error(overall_df_inner, num_iterations_cv, num_season_factors_cv, num_home_factors_cv, lam_cv):
	print num_iterations_cv, num_season_factors_cv, num_home_factors_cv, lam_cv
	inner_kf = KFold(n_splits=2)
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
		L_inner = source_L[np.ix_(np.concatenate([test_inner, train_inner]), np.concatenate([test_inner, train_inner]))]

		H, A, T, Hs, As, Ts, HATs, costs = learn_HAT_adagrad_graph(case, tensor_copy_inner, L_inner, 
																	num_home_factors_cv, num_season_factors_cv, 
																	num_iter=num_iterations_cv, lr=1, dis=False, 
																	lam=lam_cv)
														


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
				err[appliance] = compute_rmse_fraction(appliance, pred_inner[appliance][range(4, 10)], source)[2]
			else:
				err[appliance] = compute_rmse_fraction(appliance, pred_inner[appliance], source)[2]
			appliance_to_weight.append(appliance)
		except Exception, e:
			# This appliance does not have enough samples. Will not be
			# weighed
			print(e)
			print(appliance)
	print "Done ", num_iterations_cv
	print("Error weighted on: {}".format(appliance_to_weight))
	err_weight = {}
	for appliance in appliance_to_weight:
		err_weight[appliance] = err[appliance] * contri[source][appliance]
	mean_err = pd.Series(err_weight).sum()
	return mean_err


pred = {}
n_splits = 10

cost = 'l21'
algo = 'adagrad'


for appliance in APPLIANCES_ORDER:
	pred[appliance] = []
print(source, random_seed, train_percentage)
best_params_global = {}
kf = KFold(n_splits=n_splits)
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
	
	best_num_iterations = 0
	best_num_season_factors = 0
	best_num_home_factors = 0
	best_appliance_wise_err = {appliance: 1e6 for appliance in APPLIANCES_ORDER[1:]}
	least_error = 1e6

	overall_df_inner = source_df.loc[train_ix]
	best_params_global[outer_loop_iteration] = {}
	params = {}
	count = 0

	##############################################################
	# Parallel part
	results = []
	cpus = mp.cpu_count()
	pool = mp.Pool()
	for num_iterations_cv in [1300, 100, 900, 500]:
		for num_season_factors_cv in range(2, 5):
			for num_home_factors_cv in range(3, 6):
				for lam_cv in [0]:
					params[count] = []
					params[count].extend((overall_df_inner, num_iterations_cv, num_season_factors_cv, num_home_factors_cv, lam_cv))
					count += 1

	for i in range(count): 
		result = pool.apply_async(compute_inner_error, params[i])
		results.append(result)
	pool.close()
	pool.join()
	# End of parallel part
	###############################################################
	#
	
	# get the results of all processes
	error = []
	for result in results:
		error.append(result.get())
	# get the parameters for the best setting
	best_idx = np.argmin(error)
	overall_df_inner, best_num_iterations, best_num_season_factors, best_num_home_factors, best_lam = params[best_idx]
	least_error = error[best_idx]


	best_params_global[outer_loop_iteration] = {'Iterations': best_num_iterations,
												'Num season factors': best_num_season_factors,
												'Num home factors': best_num_home_factors,
												'lambda': best_lam,
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

	L = source_L[np.ix_(np.concatenate([test, train]), np.concatenate([test, train]))]
	
	H, A, T, Hs, As, Ts, HATs, costs = learn_HAT_adagrad_graph(case, tensor_copy, L, 
																best_num_home_factors, best_num_season_factors, 
																num_iter=best_num_iterations, lr=1, dis=False, 
																lam=best_lam
																)




	HAT = multiply_case(H, A, T, case)
	for appliance in APPLIANCES_ORDER:
		pred[appliance].append(pd.DataFrame(HAT[:num_test, appliance_index[appliance], :], index=test_ix))

for appliance in APPLIANCES_ORDER:
	pred[appliance] = pd.DataFrame(pd.concat(pred[appliance]))

out = {'Predictions':pred, 'Learning Params':best_params_global}

name = "{}-{}".format(random_seed, train_percentage)
directory = os.path.expanduser('~/git/pred_tf/{}'.format(source))
if not os.path.exists(directory):
	os.makedirs(directory)
filename = os.path.expanduser('~/git/pred_tf/{}'.format(source)+ name + '.pkl')

if os.path.exists(filename):
	print("File already exists. Quitting.")
	#sys.exit(0)

with open(filename, 'wb') as f:
	pickle.dump(out, f, pickle.HIGHEST_PROTOCOL)

