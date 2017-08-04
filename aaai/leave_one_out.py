from sklearn.model_selection import KFold
from create_matrix import *
import os
import sys
from tensor_custom_core import *
from create_matrix import *
from tensor_custom_core import *
from degree_days import dds
from sklearn.model_selection import train_test_split, KFold
from common import compute_rmse_fraction
from common import compute_rmse
import pickle

appliance_index = {appliance: APPLIANCES_ORDER.index(appliance) for appliance in APPLIANCES_ORDER}
APPLIANCES = ['fridge', 'hvac', 'wm', 'mw', 'oven', 'dw']
year = 2014
n_splits = 10
case=2
a=2
cost='abs'

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

def save_obj(obj, name ):
	with open(os.path.expanduser('~/git/'+ name + '.pkl'), 'wb') as f:
		pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

region = 'Austin'
au_df, au_dfc = create_matrix_single_region("Austin", year)
au_tensor = get_tensor(au_df, au_dfc)
static_au = au_df[['area','total_occupants','num_rooms']].copy()
static_au['area'] = static_au['area'].div(4000)
static_au['total_occupants'] = static_au['total_occupants'].div(8)
static_au['num_rooms'] = static_au['num_rooms'].div(8)
static_au = static_au.values
region = 'SanDiego'
sd_df, sd_dfc = create_matrix_single_region("SanDiego", year)
sd_tensor = get_tensor(sd_df, sd_dfc)
static_sd = sd_df[['area','total_occupants','num_rooms']].copy()
static_sd['area'] = static_sd['area'].div(4000)
static_sd['total_occupants'] = static_sd['total_occupants'].div(8)
static_sd['num_rooms'] = static_sd['num_rooms'].div(8)
static_sd = static_sd.values


m,n,o = sd_tensor.shape
static_fac, algo, k, per= sys.argv[1:]
k = int(k)
per = int(per)

cost = 'l21'
#algo = 'adagrad'


if static_fac == 'None':
	H_known_Au = None
	H_known_Sd = None
else:
	H_known_Au = static_au
	H_known_Sd = static_sd


b = 3
if algo == 'adagrad':
	cost = 'l21'
	if static_fac == 'static':
		a = 5
		H_au, A_au, T_au, Hs, As, Ts, HATs, costs = learn_HAT_adagrad(case, au_tensor, a, b, num_iter=2000, lr=0.1, dis=False, cost_function=cost, H_known = H_known_Au, T_known=np.ones(12).reshape(-1, 1))
	else:
		a = 2
		H_au, A_au, T_au, Hs, As, Ts, HATs, costs = learn_HAT_adagrad(case, au_tensor, a, b, num_iter=2000, lr=0.1, dis=False, cost_function=cost, T_known=np.ones(12).reshape(-1, 1))
else:
	cost = 'abs'
	if static_fac == 'static':
		a = 5
		H_au, A_au, T_au = learn_HAT(case, au_tensor, a, b, num_iter=2000, lr=0.1, dis=False, cost_function=cost, H_known = H_known_Au, T_known=np.ones(12).reshape(-1, 1))
	else:
		a = 2
		H_au, A_au, T_au = learn_HAT(case, au_tensor, a, b, num_iter=2000, lr=0.1, dis=False, cost_function=cost, T_known=np.ones(12).reshape(-1, 1))

print (static_fac, algo, k)

## leave one cell out
# m,n,t = sd_tensor.shape
# pred_cell = np.empty([m,n])
# for i in range(m):
# 	for j in range(1,n):
# 		print i, j, k
# 		tensor_copy = sd_tensor.copy()
# 		tensor_copy[i, j, k] = np.NaN
		
# 		if algo == 'adagrad':
# 			cost = 'l21'
# 			if static_fac == 'static':
# 				a = 5
# 				H_sd, A_sd, T_sd, Hs, As, Ts, HATs, costs = learn_HAT_adagrad(case, tensor_copy, a, b, num_iter=2000, lr=0.1, dis=False, cost_function=cost, H_known = H_known_Sd, A_known = A_au, T_known=np.ones(12).reshape(-1, 1))
# 			else:
# 				a = 2
# 				H_sd, A_sd, T_sd, Hs, As, Ts, HATs, costs = learn_HAT_adagrad(case, tensor_copy, a, b, num_iter=2000, lr=0.1, dis=False, cost_function=cost, A_known = A_au, T_known=np.ones(12).reshape(-1, 1))
# 		else:
# 			cost = 'abs'
# 			if static_fac == 'static':
# 				a = 5
# 				H_sd, A_sd, T_sd = learn_HAT(case, tensor_copy, a, b, num_iter=2000, lr=0.1, dis=False, cost_function=cost, H_known = H_known_Sd, A_known = A_au, T_known=np.ones(12).reshape(-1, 1))
# 			else:
# 				a = 2
# 				H_sd, A_sd, T_sd = learn_HAT(case, tensor_copy, a, b, num_iter=2000, lr=0.1, dis=False, cost_function=cost, A_known = A_au, T_known=np.ones(12).reshape(-1, 1))

# 		prediction = multiply_case(H_sd,A_sd,T_sd,case)
# 		pred_cell[i][j] = prediction[i][j][k]


# save_obj(pred_cell, "pred_cell_month_" + static_fac + "_" + algo + "_" + str(k))

## leave one appliance out for test homes (cross validation)
pred_cv = {}

for appliance in APPLIANCES_ORDER:
	pred_cv[appliance] = {f:[] for f in range(10, 110, 40)}

kf = KFold(n_splits=n_splits)

for train_percentage in [per]:
	print "training percentage: ", train_percentage
	rd = 0
	for train_max, test in kf.split(sd_df):
			print "round: ", rd

			num_train = int((train_percentage*len(train_max)/100)+0.5)
			num_test = len(test)

			# get the random training data from train_max based on then random seed
			if train_percentage==100:
				train = train_max
			else:
				train, _ = train_test_split(train_max, train_size = train_percentage/100.0, random_state=k)

			# get the index of training and testing data
			train_ix = sd_df.index[train]
			test_ix = sd_df.index[test]
			print "test_ix: ", test_ix

			# create the tensor
			train_test_ix = np.concatenate([test_ix, train_ix])
			df_t, dfc_t = sd_df.ix[train_test_ix], sd_dfc.ix[train_test_ix]
			tensor = get_tensor(df_t, dfc_t)


			#############################################################################################
			# transfer learning: constant constraint, with A_known = A_a_const, with learn_HAT_constant
			############################################################################################
			pred = np.empty([num_test, 7, 12])
			
			for i in range(1, 7):
				for j in range(12):
					tensor_copy = tensor.copy()
					tensor_copy[:num_test, i, j] = np.NaN
					print i, j
					if algo == 'adagrad':
						cost = 'l21'
						if static_fac == 'static':
							a = 5
							H_sd, A_sd, T_sd, Hs, As, Ts, HATs, costs = learn_HAT_adagrad(case, tensor_copy, a, b, num_iter=2000, lr=0.1, dis=False, cost_function=cost, H_known = H_known_Sd[np.concatenate([test, train])], A_known = A_au, T_known=np.ones(12).reshape(-1, 1))
						else:
							a = 2
							H_sd, A_sd, T_sd, Hs, As, Ts, HATs, costs = learn_HAT_adagrad(case, tensor_copy, a, b, num_iter=2000, lr=0.1, dis=False, cost_function=cost, A_known = A_au, T_known=np.ones(12).reshape(-1, 1))
					else:
						cost = 'abs'
						if static_fac == 'static':
							a = 5
							H_sd, A_sd, T_sd = learn_HAT(case, tensor_copy, a, b, num_iter=2000, lr=0.1, dis=False, cost_function=cost, H_known = H_known_Sd[np.concatenate([test, train])], A_known = A_au, T_known=np.ones(12).reshape(-1, 1))
						else:
							a = 2
							H_sd, A_sd, T_sd = learn_HAT(case, tensor_copy, a, b, num_iter=2000, lr=0.1, dis=False, cost_function=cost, A_known = A_au, T_known=np.ones(12).reshape(-1, 1))


					HAT = multiply_case(H_sd, A_sd, T_sd, case)
					pred[:, i, j] = HAT[:num_test, i, j]
			# get the prediction
			
			for appliance in APPLIANCES_ORDER:
				pred_cv[appliance][train_percentage].append(pd.DataFrame(pred[:, appliance_index[appliance], :], index=test_ix))

			rd += 1
save_obj(pred_cv, "pred_month_" + static_fac + "_" + algo + "_" + str(k) + "_" + str(per))
