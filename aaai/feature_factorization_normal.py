import pickle

from sklearn.model_selection import train_test_split, KFold

from aaai18.common import compute_rmse_fraction, contri
from create_matrix import *
from tensor_custom_core import *

appliance_index = {appliance: APPLIANCES_ORDER.index(appliance) for appliance in APPLIANCES_ORDER}

APPLIANCES = ['fridge', 'hvac', 'wm', 'mw', 'oven', 'dw']

year = 2014

import os


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
	if source=="Austin":
		df = df.head(50)
		dfc = df.head(50)
	tensor = get_tensor(df)
	static_region = df[['area', 'total_occupants', 'num_rooms']].copy()
	static_region['area'] = static_region['area'].div(4000)
	static_region['total_occupants'] = static_region['total_occupants'].div(8)
	static_region['num_rooms'] = static_region['num_rooms'].div(8)
	static_region = static_region.values
	return df, dfc, tensor, static_region

def cost_feature(H, A, T, H_sub, F, E_np_masked, static, case):
    HAT = multiply_case(H, A, T, case)
    mask = ~np.isnan(E_np_masked)
    error_1 = (HAT - E_np_masked)[mask].flatten()
    
    HF = np.einsum('na,ab->nb', H_sub, F)
    mask_static = ~np.isnan(static)
    error_2 = (HF - static)[mask_static].flatten()
    
    return np.sqrt((error_1**2).mean()) + np.sqrt((error_2**2).mean())

def learn_HAT_adagrad_feature(case, E_np_masked, static, a, b, num_iter=2000, lr=0.1, dis=False, cost_function='abs', H_known=None, A_known=None, T_known=None, random_seed=0, eps=1e-8, penalty_coeff=0.0):

    cost = cost_feature
    mg = multigrad(cost, argnums=[0, 1, 2, 3, 4])

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
    H_1 = np.random.rand(params['M'], 2)
    F = np.random.rand(2,3)

    sum_square_gradients_H = np.zeros_like(H)
    sum_square_gradients_A = np.zeros_like(A)
    sum_square_gradients_T = np.zeros_like(T)
    sum_square_gradients_H_1 = np.zeros_like(H_1)
    sum_square_gradients_F = np.zeros_like(F)

    Hs = [H.copy()]
    As = [A.copy()]
    Ts = [T.copy()]
    H_1s = [H_1.copy()]
    Fs = [F.copy()]
    
    costs = [cost(H, A, T, H_1, F, E_np_masked, static, case)]
    
    HATs = [multiply_case(H, A, T, 2)]

    # GD procedure
    for i in range(num_iter):
        del_h, del_a, del_t, del_h_1, del_f = mg(H, A, T, H_1, F, E_np_masked, static, case)
        sum_square_gradients_H += eps + np.square(del_h)
        sum_square_gradients_A += eps + np.square(del_a)
        sum_square_gradients_T += eps + np.square(del_t)
        sum_square_gradients_H_1 += eps + np.square(del_h_1)
        sum_square_gradients_F += eps + np.square(del_f)

        lr_h = np.divide(lr, np.sqrt(sum_square_gradients_H))
        lr_a = np.divide(lr, np.sqrt(sum_square_gradients_A))
        lr_t = np.divide(lr, np.sqrt(sum_square_gradients_T))
        lr_h_1 = np.divide(lr, np.sqrt(sum_square_gradients_H_1))
        lr_f = np.divide(lr, np.sqrt(sum_square_gradients_F))

        H -= lr_h * del_h
        A -= lr_a * del_a
        T -= lr_t * del_t
        H_1 -= lr_h_1 * del_h_1
        F -= lr_f * del_f
        # Projection to known values
        if H_known is not None:
            H = set_known(H, H_known)
        if A_known is not None:
            A = set_known(A, A_known)
        if T_known is not None:
            T = set_known(T, T_known)
        H = set_known(H, H_1)
        # Projection to non-negative space
        H[H < 0] = 1e-8
        A[A < 0] = 1e-8
        T[T < 0] = 1e-8
        H_1[H_1 < 0] = 1e-8
        F[F < 0] = 1e-8

        As.append(A.copy())
        Ts.append(T.copy())
        Hs.append(H.copy())
        H_1s.append(H_1.copy())
        Fs.append(F.copy())
        
        costs.append(cost(H, A, T, H_1, F, E_np_masked, static, case))
        
        HATs.append(multiply_case(H, A, T, 2))
        if i % 500 == 0:
            if dis:
                print(cost(H, A, T, H_1, F, E_np_masked, static, case))
    return H, A, T, H_1, F, Hs, As, Ts, H_1s, Fs, HATs, costs

n_splits = 10
case = 2

feature_fac, random_seed, train_percentage = sys.argv[1:]
random_seed = int(random_seed)
train_percentage = float(train_percentage)

source = 'SanDiego'
source_df, source_dfc, source_tensor, source_static = create_region_df_dfc_static(source, year)

pred = {}
sd = {}
out = {}
n_splits = 10
case = 2
n_iter = 1200

cost = 'l21'
algo = 'adagrad'


np.random.seed(random_seed)
kf = KFold(n_splits=n_splits)
pred = {}

for appliance in APPLIANCES_ORDER:
	pred[appliance] = []
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
	for num_iterations_cv in range(1000, 2200, 1000):
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
					if feature_fac == 'feature':
						H, A, T, H_feature, F, Hs, As, Ts, H_features, Fs, HATs, costs = learn_HAT_adagrad_feature(case, tensor_copy_inner, source_static[np.concatenate([test_inner, train_inner])],
                                                                                                            num_home_factors_cv, num_season_factors_cv,
                                                                                                         num_iter=num_iterations_cv, lr=0.1, dis=False,
                                                                                                         T_known=np.ones(12).reshape(-1,1))
                        
					else:
						H, A, T, Hs, As, Ts, HATs, costs = learn_HAT_adagrad(case, tensor_copy_inner,
						                                                     num_home_factors_cv,
						                                                     num_season_factors_cv,
						                                                     num_iter=num_iterations_cv, lr=0.1,
						                                                     dis=False,
						                                                     cost_function=cost,
						                                                     H_known=source_static[np.concatenate([test_inner, train_inner])],
						                                                     T_known=np.ones(12).reshape(-1,1)
						                                                     )
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
								compute_rmse_fraction(appliance, pred_inner[appliance][range(4, 10)], source)[2]
						else:
							err[appliance] = compute_rmse_fraction(appliance, pred_inner[appliance], source)[2]
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
	if feature_fac == 'feature':
		H, A, T, H_feature, F, Hs, As, Ts, H_features, Fs, HATs, costs = learn_HAT_adagrad_feature(case, tensor_copy, source_static[np.concatenate([test, train])],
                                                                                                          best_num_home_factors, best_num_season_factors,
                                                                                                         num_iter=best_num_iterations, lr=0.1, dis=False,
                                                                                                         T_known=np.ones(12).reshape(-1,1))
                        
	else:
		H, A, T, Hs, As, Ts, HATs, costs = learn_HAT_adagrad(case, tensor_copy,
		                                                     best_num_home_factors,
		                                                     best_num_season_factors,
		                                                     num_iter=best_num_iterations, lr=0.1,
		                                                     dis=False,
		                                                     cost_function=cost,
		                                                     H_known=source_static[np.concatenate([test, train])],
		                                                     T_known=np.ones(12).reshape(-1,1)
		                                                     )

	HAT = multiply_case(H, A, T, case)
	for appliance in APPLIANCES_ORDER:
		pred[appliance].append(pd.DataFrame(HAT[:num_test, appliance_index[appliance], :], index=test_ix))

for appliance in APPLIANCES_ORDER:
	pred[appliance] = pd.DataFrame(pd.concat(pred[appliance]))

name = "{}-{}-{}".format(feature_fac, random_seed, train_percentage)
directory = os.path.expanduser('~/git/pred_feature/normal_{}/'.format(feature_fac))
if not os.path.exists(directory):
    os.makedirs(directory)
filename = os.path.expanduser('~/git/pred_feature/normal_{}/'.format(feature_fac) + name + '.pkl')

if os.path.exists(filename):
    print("File already exists. Quitting.")

out = {'Predictions': pred, 'Learning Params': best_params_global}
with open(filename, 'wb') as f:
	pickle.dump(out, f, pickle.HIGHEST_PROTOCOL)
