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
	nbrs = NearestNeighbors(n_neighbors=5, radius=0.05, algorithm='ball_tree').fit(X)
	distances, indices = nbrs.kneighbors(X)
	n_sample, n_feature = X.shape
	W = np.zeros((n_sample, n_sample))
	for i in range(n_sample):
		if distances[i][4] == 0:
			continue
		for j in indices[i]:
			W[i][j] = 1
			W[j][i] = 1
	K = np.dot(W, np.ones((n_sample, n_sample)))
	D = np.diag(np.diag(K))
	return D - W


def get_L(X):
	W = 1 - squareform(pdist(X, 'cosine'))
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

	return np.sqrt((error_1 ** 2).mean()) + lam * error_2


def learn_HAT_adagrad_graph(case, E_np_masked, L, a, b, num_iter=2000, lr=0.01, dis=False, lam=1, H_known=None,
                            A_known=None, T_known=None, random_seed=0, eps=1e-8, penalty_coeff=0.0):
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




source = sys.argv[1]
source_df, source_dfc, source_tensor, source_static = create_region_df_dfc_static(source, year)

# # using cosine similarity to compute L
source_L = get_L(source_static)

pred = {}
n_splits = 10
case = 2

algo = 'adagrad'
cost = 'l21'

for appliance in APPLIANCES_ORDER:
	pred[appliance] = []
best_params_global = {}
A_store = {}

max_num_iterations = 1300
for num_season_factors_cv in range(2, 5):

	A_store[num_season_factors_cv] = {}
	for num_home_factors_cv in range(3, 6):
		A_store[num_season_factors_cv][num_home_factors_cv] = {}
		for lam_cv in [0]:
			A_store[num_season_factors_cv][num_home_factors_cv][lam_cv] = {}

			H_source, A_source, T_source, Hs, As, Ts, HATs, costs = learn_HAT_adagrad_graph(case, source_tensor,
			                                                                                source_L,
			                                                                                num_home_factors_cv,
			                                                                                num_season_factors_cv,
			                                                                                num_iter=max_num_iterations,
			                                                                                lr=1, dis=False,
			                                                                                lam=lam_cv)
			for num_iterations in range(100, 1400, 200):
				A_store[num_season_factors_cv][num_home_factors_cv][lam_cv][num_iterations] = As[num_iterations]
				print(num_season_factors_cv, num_home_factors_cv, lam_cv, num_iterations)

pickle.dump(A_store, open('predictions/tf_{}_As.pkl'.format(source), 'w'))