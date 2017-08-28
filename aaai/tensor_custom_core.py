import autograd.numpy as np
from autograd import multigrad

cases = {
	1: {'HA': 'Ma, Nb -> MNab', 'HAT': 'MNab, Oab -> MNO'},
	2: {'HA': 'Ma, Nab -> MNb', 'HAT': 'MNb, Ob -> MNO'},
	3: {'HA': 'Mab, Na -> MNb', 'HAT': 'MNb, Ob -> MNO'},
	4: {'HA': 'Ma, Na -> MNa', 'HAT': 'MNa, Oa -> MNO'}
}


def multiply_case(H, A, T, case):
	HA = np.einsum(cases[case]['HA'], H, A)
	HAT = np.einsum(cases[case]['HAT'], HA, T)
	return HAT


def cost_abs(H, A, T, E_np_masked, case):
	HAT = multiply_case(H, A, T, case)
	mask = ~np.isnan(E_np_masked)
	error = (HAT - E_np_masked)[mask].flatten()
	return np.sqrt((error ** 2).mean())

def cost_fraction(H, A, T, E_np_masked, case):
	HAT = multiply_case(H, A, T, case)
	num_appliances = len(A)-1
	c = 0
	for appliance_num in range(1, num_appliances + 1):
		gt_appliance_fr = E_np_masked[:, appliance_num, :] / E_np_masked[:, 0, :]
		pred_appliance_fr = HAT[:, appliance_num, :] / E_np_masked[:, 0, :]
		diff_appliance_fr = (pred_appliance_fr - gt_appliance_fr).flatten()
		diff_appliance_fr = diff_appliance_fr[~np.isnan(diff_appliance_fr)]
		c = c + np.sqrt(np.square(diff_appliance_fr).mean())
	return c


def cost_rel_per(H, A, T, E_np_masked, case):
	HAT = multiply_case(H, A, T, case)
	mask = ~np.isnan(E_np_masked)
	error = 100*(HAT - E_np_masked)[mask].flatten() / (1 + E_np_masked[mask].flatten())
	return np.sqrt((error ** 2).mean())



def cost_rel(H, A, T, E_np_masked, case):
	HAT = multiply_case(H, A, T, case)
	mask = ~np.isnan(E_np_masked)
	error = (HAT - E_np_masked)[mask].flatten() / (1 + E_np_masked[mask].flatten())
	return np.sqrt((error ** 2).mean())


def set_known(A, W):
	mask = ~np.isnan(W)
	A[:, :mask.shape[1]][mask] = W[mask]
	return A


def learn_HAT_multiple_source_adagrad(case, source_1_energy, source_2_energy, a, b, num_iter=2000, lr=0.1, dis=False,  H_known_s1=None,
                      A_known=None, T_known_s1=None, H_known_s2=None, T_known_s2 = None,
                        random_seed=0, eps=1e-8, penalty_coeff=0.0, source_ratio=0.5):





	def cost_l21(H_s1, A, T_s1, H_s2, T_s2, source_1_energy, source_2_energy, case, source_ratio, lam=0.1):
		HAT_s1 = multiply_case(H_s1, A, T_s1, case)
		HAT_s2 = multiply_case(H_s2, A, T_s2, case)
		mask_s1 = ~np.isnan(source_1_energy)
		mask_s2 = ~np.isnan(source_2_energy)
		error_s1 = (HAT_s1 - source_1_energy)[mask_s1].flatten()
		error_s2 = (HAT_s2 - source_2_energy)[mask_s2].flatten()
		A_shape = A.shape
		A_flat = A.reshape(A_shape[0], A_shape[1]*A_shape[2])
		l1 = 0.
		for j in range(A_shape[1]*A_shape[2]):
			l1 = l1 + np.sqrt(np.square(A_flat[:, j]).sum())
		# return np.sqrt((error ** 2).mean()) + lam*np.sum(A[A!=0])
		return source_ratio*np.sqrt((error_s1 ** 2).mean()) + (1-source_ratio)*np.sqrt((error_s2 ** 2).mean())+ lam * l1



	np.random.seed(random_seed)

	cost = cost_l21


	mg = multigrad(cost, argnums=[0, 1, 2, 3, 4])

	params_s1 = {}
	params_s1['M'], params_s1['N'], params_s1['O'] = source_1_energy.shape
	params_s1['a'] = a
	params_s1['b'] = b

	params_s2 = {}
	params_s2['M'], params_s2['N'], params_s2['O'] = source_2_energy.shape
	params_s2['a'] = a
	params_s2['b'] = b

	H_dim_chars = list(cases[case]['HA'].split(",")[0].strip())
	H_dim_s1 = tuple(params_s1[x] for x in H_dim_chars)
	H_dim_s2 = tuple(params_s2[x] for x in H_dim_chars)

	A_dim_chars = list(cases[case]['HA'].split(",")[1].split("-")[0].strip())
	A_dim_s1 = tuple(params_s1[x] for x in A_dim_chars)
	A_dim_s2 = tuple(params_s2[x] for x in A_dim_chars)

	T_dim_chars = list(cases[case]['HAT'].split(",")[1].split("-")[0].strip())
	T_dim_s1 = tuple(params_s1[x] for x in T_dim_chars)
	T_dim_s2 = tuple(params_s2[x] for x in T_dim_chars)


	H_s1 = np.random.rand(*H_dim_s1)
	H_s2 = np.random.rand(*H_dim_s2)

	A = np.random.rand(*A_dim_s1)
	T_s1 = np.random.rand(*T_dim_s1)
	T_s2 = np.random.rand(*T_dim_s2)

	sum_square_gradients_H_s1 = np.zeros_like(H_s1)
	sum_square_gradients_H_s2 = np.zeros_like(H_s2)

	sum_square_gradients_A = np.zeros_like(A)

	sum_square_gradients_T_s1 = np.zeros_like(T_s1)
	sum_square_gradients_T_s2 = np.zeros_like(T_s2)


	Hs_s1 = [H_s1.copy()]
	Hs_s2 = [H_s2.copy()]
	As = [A.copy()]
	Ts_s1 = [T_s1.copy()]
	Ts_s2 = [T_s2.copy()]
	costs = [cost(H_s1, A, T_s1, H_s2, T_s2, source_1_energy, source_2_energy, 2,  penalty_coeff, source_ratio)]
	HATs_s1 = [multiply_case(H_s1, A, T_s1, 2)]
	HATs_s2 = [multiply_case(H_s2, A, T_s2, 2)]

	# GD procedure
	for i in range(num_iter):
		del_h_s1, del_a, del_t_s1, del_h_s2, del_t_s2 = mg(H_s1, A, T_s1, H_s2, T_s2, source_1_energy, source_2_energy, case, source_ratio, penalty_coeff)
		sum_square_gradients_H_s1 += eps + np.square(del_h_s1)
		sum_square_gradients_H_s2 += eps + np.square(del_h_s2)

		sum_square_gradients_A += eps + np.square(del_a)

		sum_square_gradients_T_s1 += eps + np.square(del_t_s1)
		sum_square_gradients_T_s2 += eps + np.square(del_t_s2)

		lr_h_s1 = np.divide(lr, np.sqrt(sum_square_gradients_H_s1))
		lr_h_s2 = np.divide(lr, np.sqrt(sum_square_gradients_H_s2))

		lr_a = np.divide(lr, np.sqrt(sum_square_gradients_A))

		lr_t_s1 = np.divide(lr, np.sqrt(sum_square_gradients_T_s1))
		lr_t_s2 = np.divide(lr, np.sqrt(sum_square_gradients_T_s2))

		H_s1 -= lr_h_s1 * del_h_s1
		H_s2 -= lr_h_s2 * del_h_s2
		A -= lr_a * del_a

		T_s1 -= lr_t_s1 * del_t_s1
		T_s2 -= lr_t_s2 * del_t_s2
		# Projection to known values
		if H_known_s1 is not None:
			H_s1 = set_known(H_s1, H_known_s1)
		if H_known_s2 is not None:
			H_s2 = set_known(H_s2, H_known_s2)
		if A_known is not None:
			A = set_known(A, A_known)
		if T_known_s1 is not None:
			T = set_known(T, T_known_s1)
		if T_known_s2 is not None:
			T = set_known(T, T_known_s2)
		# Projection to non-negative space
		H_s1[H_s1 < 0] = 1e-8
		H_s2[H_s2 < 0] = 1e-8
		A[A < 0] = 1e-8
		T_s1[T_s1 < 0] = 1e-8
		T_s2[T_s2 < 0] = 1e-8

		As.append(A.copy())
		Ts_s1.append(T_s1.copy())
		Ts_s2.append(T_s2.copy())
		Hs_s1.append(H_s1.copy())
		Hs_s2.append(H_s2.copy())
		costs.append(cost(H_s1, A, T_s1, H_s2, T_s2, source_1_energy, source_2_energy, 2,  penalty_coeff,source_ratio))
		HATs_s1.append(multiply_case(H_s1, A, T_s1, 2))
		HATs_s2.append(multiply_case(H_s2, A, T_s2, 2))
		if i % 500 == 0:
			if dis:
				print(cost(H_s1, A, T_s1, H_s2, T_s2, source_1_energy, source_2_energy, 2,  penalty_coeff, source_ratio))
	return H_s1, A, T_s1, H_s2, T_s2, Hs_s1, As, Ts_s1, Hs_s2,  Ts_s2, HATs_s1, HATs_s2, costs



def learn_HAT_adagrad(case, E_np_masked, a, b, num_iter=2000, lr=0.1, dis=False, cost_function='abs', H_known=None,
                      A_known=None, T_known=None, random_seed=0, eps=1e-8, penalty_coeff=0.0, non_neg=True):


	def cost_l12(H, A, T, E_np_masked, case, lam=0.1):
		HAT = multiply_case(H, A, T, case)
		mask = ~np.isnan(E_np_masked)
		error = (HAT - E_np_masked)[mask].flatten()
		A_shape = A.shape
		A_flat = A.reshape(A_shape[0], A_shape[1]*A_shape[2])
		l1 = 0.
		for j in range(A_shape[0]):
			l1 = l1 + np.sqrt(np.square(A_flat[j,:]).sum())
			#print(j, l1)
		# return np.sqrt((error ** 2).mean()) + lam*np.sum(A[A!=0])
		return np.sqrt((error ** 2).mean()) + lam * l1

	def cost_l1(H, A, T, E_np_masked, case, lam=0.1):
		HAT = multiply_case(H, A, T, case)
		mask = ~np.isnan(E_np_masked)
		error = (HAT - E_np_masked)[mask].flatten()
		A_shape = A.shape
		A_flat = A.reshape(A_shape[0]*A_shape[1] * A_shape[2], )

		return np.sqrt((error ** 2).mean()) + lam * np.sum(A_flat)

	def cost_l1_without_flat(H, A, T, E_np_masked, case, lam=0.1):
		HAT = multiply_case(H, A, T, case)
		mask = ~np.isnan(E_np_masked)
		error = (HAT - E_np_masked)[mask].flatten()


		return np.sqrt((error ** 2).mean()) + lam * np.sum(A)



	def cost_l21(H, A, T, E_np_masked, case, lam=0.1):
		HAT = multiply_case(H, A, T, case)
		mask = ~np.isnan(E_np_masked)
		error = (HAT - E_np_masked)[mask].flatten()
		A_shape = A.shape
		A_flat = A.reshape(A_shape[0], A_shape[1]*A_shape[2])
		l1 = 0.
		for j in range(A_shape[1]*A_shape[2]):
			l1 = l1 + np.sqrt(np.square(A_flat[:, j]).sum())
		# return np.sqrt((error ** 2).mean()) + lam*np.sum(A[A!=0])
		return np.sqrt((error ** 2).mean()) + lam * l1

	def cost_abs_penalty_sum_squares(H, A, T, E_np_masked, case, lam=0.1):
		HAT = multiply_case(H, A, T, case)
		mask = ~np.isnan(E_np_masked)
		error = (HAT - E_np_masked)[mask].flatten()
		# return np.sqrt((error ** 2).mean()) + lam*np.sum(A[A!=0])
		return np.sqrt((error ** 2).mean()) + lam * np.sum(np.square(A))

	def cost_abs_penalty_sum_abs(H, A, T, E_np_masked, case, lam=0.1):
		HAT = multiply_case(H, A, T, case)
		mask = ~np.isnan(E_np_masked)
		error = (HAT - E_np_masked)[mask].flatten()
		# return np.sqrt((error ** 2).mean()) + lam*np.sum(A[A!=0])
		return np.sqrt((error ** 2).mean()) + lam * np.sum(np.abs(A))

	def cost_abs_penalty_count_non_zero(H, A, T, E_np_masked, case, lam=0.1):
		HAT = multiply_case(H, A, T, case)
		mask = ~np.isnan(E_np_masked)
		error = (HAT - E_np_masked)[mask].flatten()
		# return np.sqrt((error ** 2).mean()) + lam*np.sum(A[A!=0])
		return np.sqrt((error ** 2).mean()) + lam * np.count_nonzero(A)


	np.random.seed(random_seed)
	if cost_function == 'abs':
		cost = cost_abs
	elif cost_function =='rel':
		cost = cost_rel
	elif cost_function =='fraction':
		cost = cost_fraction
	elif cost_function =='penalty-sum-squares':
		cost = cost_abs_penalty_sum_squares
	elif cost_function =='penalty-sum-abs':
		cost = cost_abs_penalty_sum_abs
	elif cost_function == 'penalty-count-nonzero':
		cost = cost_abs_penalty_count_non_zero
	elif cost_function == 'l21':
		cost = cost_l21
	elif cost_function == 'l12':
		cost = cost_l12
	elif cost_function == 'l1':
		cost = cost_l1
	elif cost_function == 'l1-without-flat':
		cost = cost_l1_without_flat


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
	if 'penalty' not in cost_function:
		costs = [cost(H, A, T, E_np_masked, 2)]
	else:
		costs = [cost(H, A, T, E_np_masked, 2, penalty_coeff)]
	HATs = [multiply_case(H, A, T, 2)]

	# GD procedure
	for i in range(num_iter):
		del_h, del_a, del_t = mg(H, A, T, E_np_masked, case, penalty_coeff)
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
		if non_neg:
			H[H < 0] = 1e-8
			A[A < 0] = 1e-8
			T[T < 0] = 1e-8

		As.append(A.copy())
		Ts.append(T.copy())
		Hs.append(H.copy())
		if 'penalty' not in cost_function:
			costs.append(cost(H, A, T, E_np_masked, 2))
		else:
			costs.append(cost(H, A, T, E_np_masked, 2, penalty_coeff))
		HATs.append(multiply_case(H, A, T, 2))
		if i % 500 == 0:
			if dis:
				print(cost(H, A, T, E_np_masked, case, penalty_coeff))
	return H, A, T, Hs, As, Ts, HATs, costs

def learn_HAT(case, E_np_masked, a, b, num_iter=2000, lr=0.1, dis=False, cost_function='abs', H_known=None,
              A_known=None, T_known=None, random_seed=0, decay_mul=1, batchsize=None, aggregate_constraint=False):

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



		if aggregate_constraint:
			# Projection to ensure A[aggregate] >=sum(A[appliances]
			A[0] = np.maximum(A[0], np.sum(A[1:], axis=0))

		if i % 500 == 0:
			if dis:
				print(cost(H, A, T, E_np_masked, case), lrs[i], i)
	return H, A, T

def learn_HAT_random_normal(case, E_np_masked, a, b, num_iter=2000, lr=0.1, dis=False, cost_function='abs', H_known=None,
              A_known=None, T_known=None, random_seed=0, random_mul_constant=1,
              random_add_constant=0, scale_random=20):
	np.random.seed(random_seed)
	if cost_function == 'abs':
		cost = cost_abs
	else:
		cost = cost_rel
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
	H = np.abs(np.random.normal(loc=0.0, scale=scale_random, size=H_dim))

	A = np.abs(np.random.normal(loc=0.0, scale=scale_random, size=A_dim))
	T = np.abs(np.random.normal(loc=0.0, scale=scale_random,size=T_dim))

	Hs =[H.copy()]
	As= [A.copy()]
	Ts = [T.copy()]
	costs = [cost_abs(H, A, T, E_np_masked, 2)]
	HATs =[multiply_case(H, A, T, 2)]
	# GD procedure
	for i in range(num_iter):
		del_h, del_a, del_t = mg(H, A, T, E_np_masked, case)
		H -= lr * del_h
		A -= lr * del_a
		T -= lr * del_t
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
		As.append(A.copy())
		Ts.append(T.copy())
		Hs.append(H.copy())
		costs.append(cost_abs(H, A, T, E_np_masked, 2))
		HATs.append(multiply_case(H, A, T, 2))
		if i % 500 == 0:
			if dis:
				print(cost(H, A, T, E_np_masked, case))
	return H, A, T, Hs, As, Ts, HATs, costs

