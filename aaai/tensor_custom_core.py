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


def cost_rel(H, A, T, E_np_masked, case):
	HAT = multiply_case(H, A, T, case)
	mask = ~np.isnan(E_np_masked)
	error = (HAT - E_np_masked)[mask].flatten() / (1 + E_np_masked[mask].flatten())
	return np.sqrt((error ** 2).mean())


def set_known(A, W):
	mask = ~np.isnan(W)
	A[:, :mask.shape[1]][mask] = W[mask]
	return A


def learn_HAT(case, E_np_masked, a, b, num_iter=2000, lr=0.1, dis=False, cost_function='abs', H_known=None,
              A_known=None, T_known=None, random_seed=0, random_mul_constant=1,
              random_add_constant=0):
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
	H = np.random.rand(*H_dim)*random_mul_constant+random_add_constant

	A = np.random.rand(*A_dim)*random_mul_constant+random_add_constant
	T = np.random.rand(*T_dim)*random_mul_constant+random_add_constant

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
		if i % 500 == 0:
			if dis:
				print(cost(H, A, T, E_np_masked, case))
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

	Hs =[H]
	As= [A]
	Ts = [T]
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
		As.append(A)
		Ts.append(T)
		Hs.append(H)
		costs.append(cost_abs(H, A, T, E_np_masked, 2))
		HATs.append(multiply_case(H, A, T, 2))
		if i % 500 == 0:
			if dis:
				print(cost(H, A, T, E_np_masked, case))
	return H, A, T, Hs, As, Ts, HATs, costs

