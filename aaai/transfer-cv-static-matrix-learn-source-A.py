from create_matrix import *
from tensor_custom_core import *

appliance_index = {appliance: APPLIANCES_ORDER.index(appliance) for appliance in APPLIANCES_ORDER}




APPLIANCES = ['fridge', 'hvac', 'wm', 'mw', 'oven', 'dw']

year = 2014


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

source, random_seed= sys.argv[1:]



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
	static_region =static_region.values
	return df, dfc, tensor, static_region

source_df, source_dfc, source_tensor, source_static = create_region_df_dfc_static(source, year)

pred = {}
sd = {}
out = {}
n_splits = 10
case = 2

algo = 'adagrad'


random_seed = int(random_seed)

H_known_source = source_static

np.random.seed(random_seed)


params = {source:{}}

count = 0
for num_iterations_cv in range(100, 1400, 400):
	params[source][num_iterations_cv] = {}
	for num_season_factors_cv in range(2, 5):
		params[source][num_iterations_cv][num_season_factors_cv] = {}
		for num_home_factors_cv in range(3, 6):
			params[source][num_iterations_cv][num_season_factors_cv][num_home_factors_cv]={}
			for lr_cv in [0.1, 1.]:
				params[source][num_iterations_cv][num_season_factors_cv][num_home_factors_cv][lr_cv] = {}
				for beta_cv in [1e-3,  1e-1, 0.]:
					count += 1
					print(count, num_iterations_cv, num_home_factors_cv, num_season_factors_cv, lr_cv, beta_cv)
					H_source, A_source, T_source, F_source, Hs_source, As_source, Ts_source, HATs_source, costs_source = learn_HAT_adagrad_static(case=case, E_np_masked=source_tensor, K=source_static,
						                                                                           a=num_home_factors_cv, b=num_season_factors_cv, num_iter=num_iterations_cv, lr=lr_cv, dis=False,
						                                                                           beta=beta_cv)

					params[source][num_iterations_cv][num_season_factors_cv][num_home_factors_cv][lr_cv][beta_cv] = {'A':A_source, 'H':H_source, 'T':T_source}
