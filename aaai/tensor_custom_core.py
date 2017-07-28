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


def learn_HAT_adagrad(case, E_np_masked, a, b, num_iter=2000, lr=0.1, dis=False, cost_function='abs', H_known=None,
                      A_known=None, T_known=None, random_seed=0, eps=1e-8, penalty_coeff=0.0):

	def cost_l21(H, A, T, E_np_masked, case, lam=0.1):
		HAT = multiply_case(H, A, T, case)
		mask = ~np.isnan(E_np_masked)
		error = (HAT - E_np_masked)[mask].flatten()
		# return np.sqrt((error ** 2).mean()) + lam*np.sum(A[A!=0])
		return np.sqrt((error ** 2).mean()) + lam * np.sum(np.square(A))

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
		H[H < 0] = 0
		A[A < 0] = 0
		T[T < 0] = 0

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

