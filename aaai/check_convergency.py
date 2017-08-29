from create_matrix import *
import os
import pickle

from create_matrix import *
from degree_days import dds
from tensor_custom_core import *


def un_normalize(x, maximum, minimum):
    return (maximum - minimum) * x + minimum

def get_tensor(df, dfc):
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
    # df = (1.0 * (df - col_min)) / (col_max - col_min)
    tensor = df.values.reshape((len(df), 7, stop - start))
    M, N, O = tensor.shape
    return tensor


def learn_HAT_series(case, E_np_masked, a, b, num_iter=2000, lr=0.1, dis=False, cost_function='abs', H_known=None, A_known=None, T_known=None, random_seed=0, decay_mul=1, batchsize=None, aggregate_constraint=False):

	np.random.seed(random_seed)
	lrs = lr*np.power(decay_mul, range(num_iter))
	if cost_function == 'abs':
		cost = cost_abs
	elif cost_function =='rel':
		cost = cost_rel
	elif cost_function =='fraction':
		cost = cost_fraction
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
	H = np.abs(np.random.rand(*H_dim))
	A = np.abs(np.random.rand(*A_dim))
	T = np.abs(np.random.rand(*T_dim))

	if batchsize is None:
		batchsize = len(E_np_masked)

	indices_home = range(params['M'])

	Hs = [H.copy()]
	As = [A.copy()]
	Ts = [T.copy()]
	costs = [cost(H, A, T, E_np_masked, 2)]
	HATs = [multiply_case(H, A, T, 2)]


	# GD procedure
	for i in range(num_iter):
		if batchsize < len(E_np_masked):
			indices_select = np.random.choice(indices_home, batchsize)
			del_h, del_a, del_t = mg(H[indices_select], A, T, E_np_masked[indices_select], case)
			H[indices_select] -= lrs[i] * del_h
		else:
			del_h, del_a, del_t = mg(H, A, T, E_np_masked, case)
			H -= lrs[i] * del_h
		A -= lrs[i] * del_a
		T -= lrs[i] * del_t
		# Projection to known values
		if H_known is not None:
			H = set_known(H, H_known)
		if A_known is not None:
			A = set_known(A, A_known)
		if T_known is not None:
			T = set_known(T, T_known)
		# Projection to non-negative space
		H[H < 0] = 0
		A[A < 0] = 0
		T[T < 0] = 0

		if i%250 == 0:
			print i
			Hs.append(H.copy())
			As.append(A.copy())
			Ts.append(T.copy())

			costs.append(cost(H, A, T, E_np_masked, 2))
			HATs.append(multiply_case(H, A, T, 2))


		if aggregate_constraint:
			# Projection to ensure A[aggregate] >=sum(A[appliances]
			A[0] = np.maximum(A[0], np.sum(A[1:], axis=0))

		if i % 500 == 0:
			if dis:
				print(cost(H, A, T, E_np_masked, case), lrs[i], i)
	return H, A, T, Hs, As, Ts, costs, HATs


def learn_HAT_adagrad_series(case, E_np_masked, a, b, num_iter=2000, lr=0.1, dis=False, cost_function='abs', H_known=None, A_known=None, T_known=None, random_seed=0, eps=1e-8):
	np.random.seed(random_seed)
	if cost_function == 'abs':
		cost = cost_abs
	elif cost_function =='rel':
		cost = cost_rel
	elif cost_function =='fraction':
		cost = cost_fraction
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

	Hs = [H.copy()]
	As = [A.copy()]
	Ts = [T.copy()]
	costs = [cost(H, A, T, E_np_masked, 2)]
	HATs = [multiply_case(H, A, T, 2)]

	sum_square_gradients_H = np.zeros_like(H)
	sum_square_gradients_A = np.zeros_like(A)
	sum_square_gradients_T = np.zeros_like(T)

	# GD procedure
	for i in range(num_iter):
		del_h, del_a, del_t = mg(H, A, T, E_np_masked, case)
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
		H[H < 0] = 0
		A[A < 0] = 0
		T[T < 0] = 0

		if i%250 == 0:
			print i
			Hs.append(H.copy())
			As.append(A.copy())
			Ts.append(T.copy())
			costs.append(cost(H, A, T, E_np_masked, 2))
			HATs.append(multiply_case(H, A, T, 2))

		if i % 500 == 0:
			if dis:
				print(cost(H, A, T, E_np_masked, case))
	return H, A, T, Hs, As, Ts, costs, HATs

def save_obj(obj, name ):
    with open(os.path.expanduser('~/git/'+ name + '.pkl'), 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

appliance_index = {appliance: APPLIANCES_ORDER.index(appliance) for appliance in APPLIANCES_ORDER}
APPLIANCES = ['fridge', 'hvac', 'wm', 'mw', 'oven', 'dw']
region = "SanDiego"
year = 2014
case = 2

sd_df, sd_dfc = create_matrix_single_region("SanDiego", year)
sd_tensor = get_tensor(sd_df, sd_dfc)
au_df, au_dfc = create_matrix_single_region("Austin", year)
au_tensor = get_tensor(au_df, au_dfc)

region, algo, cost, a, lr = sys.argv[1:]
a = int(a)
lr = float(lr) 

if region == 'Austin':
	tensor = au_tensor.copy()
elif region == 'SanDiego':
	tensor = sd_tensor.copy()

print region, algo, cost, a, lr

iterations = 30000

if algo == 'gd':
	H, A, T, Hs, As, Ts, costs, HATs = learn_HAT_series(case, tensor, a, a, num_iter=iterations, lr=lr, dis=False, cost_function=cost)
elif algo == 'gd_decay':
	H, A, T, Hs, As, Ts, costs, HATs = learn_HAT_series(case, tensor, a, a, num_iter=iterations, lr=lr, dis=False, cost_function=cost, decay_mul=0.995)
else:
	H, A, T, Hs, As, Ts, costs, HATs = learn_HAT_adagrad_series(case, tensor, a, a, num_iter=iterations, lr=lr, dis=False, cost_function=cost)


save_obj(Hs, "Hs_normal_" + region + "_" + algo + "_" + cost + "_" + str(a) + "_"  + str(lr))
save_obj(As, "As_normal_" + region + "_" + algo + "_" + cost + "_" + str(a) + "_"  + str(lr))
save_obj(Ts, "Ts_normal_" + region + "_" + algo + "_" + cost + "_" + str(a) + "_"  + str(lr))
save_obj(costs, "costs_normal_" + region + "_" + algo + "_" + cost + "_" + str(a) + "_"  + str(lr))
save_obj(HATs, "HATs_normal_" + region + "_" + algo + "_" + cost + "_" + str(a) + "_"  + str(lr))


if region == 'Austin':
	if algo == 'gd':
		H, A, T, Hs, As, Ts, costs, HATs = learn_HAT_series(case, sd_tensor, a, a, num_iter=iterations, lr=lr, dis=False, cost_function=cost, A_known = A)
	elif algo == 'gd_decay':
		H, A, T, Hs, As, Ts, costs, HATs = learn_HAT_series(case, sd_tensor, a, a, num_iter=iterations, lr=lr, dis=False, cost_function=cost, A_known = A, decay_mul=0.995)
	else:
		H, A, T, Hs, As, Ts, costs, HATs = learn_HAT_adagrad_series(case, sd_tensor, a, a, num_iter=iterations, lr=lr, dis=False, cost_function=cost, A_known = A)

	save_obj(Hs, "Hs_transfer_SanDiego_" + algo + "_" + cost + "_" + str(a) + "_"  + str(lr))
	save_obj(As, "As_transfer_SanDiego_" + algo + "_" + cost + "_" + str(a) + "_"  + str(lr))
	save_obj(Ts, "Ts_transfer_SanDiego_" + algo + "_" + cost + "_" + str(a) + "_"  + str(lr))
	save_obj(costs, "costs_transfer_SanDiego_" + algo + "_" + cost + "_" + str(a) + "_"  + str(lr))
	save_obj(HATs, "HATs_transfer_SanDiego_" + algo + "_" + cost + "_" + str(a) + "_"  + str(lr))
