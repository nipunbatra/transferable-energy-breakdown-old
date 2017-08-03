import sys

from sklearn.model_selection import KFold

from create_matrix import *
from sklearn.model_selection import train_test_split, KFold

from tensor_custom_core import *
from create_matrix import *
from tensor_custom_core import *
from degree_days import dds

appliance_index = {appliance: APPLIANCES_ORDER.index(appliance) for appliance in APPLIANCES_ORDER}



APPLIANCES = ['fridge', 'hvac', 'wm', 'mw', 'oven', 'dw']

year = 2014

import os


def un_normalize(x, maximum, minimum):
	return (maximum - minimum) * x + minimum


n_splits = 10
case = 2

source, target, static_fac, lam, num_home_factors, num_season_factors, random_seed, train_percentage = sys.argv[1:]
name = "{}-{}-{}-{}-{}-{}-{}-{}".format(source, target, static_fac, lam, num_home_factors, num_season_factors, random_seed, train_percentage)
filename = os.path.expanduser('~/aaai2017/transfer_{}_{}/'.format(source, target) + name + '.pkl')

if os.path.exists(filename):
	sys.exit(0)


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
target_df, target_dfc, target_tensor, target_static = create_region_df_dfc_static(target, year)

pred = {}
sd = {}
out = {}
n_splits = 10
NUM_RANDOM = 3
TRAIN_SPLITS = range(10, 110, 40)
case = 2
n_iter = 1200

cost = 'l21'
algo = 'adagrad'

lam = float(lam)
num_home_factors = int(num_home_factors)
num_season_factors = int(num_season_factors)
random_seed = int(random_seed)
train_percentage = float(train_percentage)

if static_fac is 'None':
	H_known_target = None
	H_known_source = None
else:
	H_known_target = target_static
	H_known_source = source_static
np.random.seed(0)

if static_fac == 'static':
	H_source, A_source, T_source, Hs, As, Ts, HATs, costs = learn_HAT_adagrad(case, source_tensor, num_home_factors, num_season_factors,
	                                                                          num_iter=n_iter, lr=1, dis=False, cost_function=cost,
	                                                                          H_known=H_known_source, penalty_coeff=lam)
else:
	H_source, A_source, T_source, Hs, As, Ts, HATs, costs = learn_HAT_adagrad(case, source_tensor, num_home_factors, num_season_factors,
	                                                                          num_iter=n_iter, lr=1, dis=False, cost_function=cost,
	                                                                          penalty_coeff=lam)

kf = KFold(n_splits=n_splits)

pred = {}
for appliance in APPLIANCES_ORDER:
	pred[appliance] = []
print(lam, static_fac, num_season_factors, num_home_factors, random_seed, train_percentage)
for train_max, test in kf.split(target_df):

	num_train = int((train_percentage * len(train_max) / 100) + 0.5)
	if train_percentage == 100:
		train = train_max
	else:
		train, _ = train_test_split(train_max, train_size=train_percentage / 100.0, random_state=random_seed)
	train_ix = target_df.index[train]
	test_ix = target_df.index[test]

	num_test = len(test_ix)
	train_test_ix = np.concatenate([test_ix, train_ix])
	df_t, dfc_t = target_df.loc[train_test_ix], target_dfc.loc[train_test_ix]
	tensor = get_tensor(df_t)
	tensor_copy = tensor.copy()
	# First n
	tensor_copy[:num_test, 1:, :] = np.NaN
	if static_fac is not None:
		H, A, T, Hs, As, Ts, HATs, costs = learn_HAT_adagrad(case, tensor_copy, num_home_factors, num_season_factors,
		                                                     num_iter=n_iter, lr=1, dis=False, cost_function=cost,
		                                                     A_known=A_source,
		                                                     H_known=H_known_target[np.concatenate([test, train])],
		                                                     penalty_coeff=lam)
	else:
		H, A, T, Hs, As, Ts, HATs, costs = learn_HAT_adagrad(case, tensor_copy, num_home_factors, num_season_factors,
		                                                     num_iter=n_iter, lr=1, dis=False, cost_function=cost,
		                                                     A_known=A_source, penalty_coeff=lam)

	HAT = multiply_case(H, A, T, case)
	for appliance in APPLIANCES_ORDER:
		pred[appliance].append(pd.DataFrame(HAT[:num_test, appliance_index[appliance], :], index=test_ix))

for appliance in APPLIANCES_ORDER:
	pred[appliance] = pd.DataFrame(pd.concat(pred[appliance]))

import pickle

with open(filename, 'wb') as f:
	pickle.dump(pred, f, pickle.HIGHEST_PROTOCOL)
