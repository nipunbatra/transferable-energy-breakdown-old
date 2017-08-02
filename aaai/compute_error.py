import matplotlib.pyplot as plt
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


region = "SanDiego"
sd_df, sd_dfc = create_matrix_single_region("SanDiego", year)
sd_tensor = get_tensor(sd_df, sd_dfc)
region = "Austin"
au_df, au_dfc = create_matrix_single_region("Austin", year)
au_tensor = get_tensor(au_df, au_dfc)

def load_obj(name):
	with open(os.path.expanduser('~/git/pred_adagrad/' + name + '.pkl'), 'rb') as f:
		return pickle.load(f)
def save_obj(obj, name ):
	with open(os.path.expanduser('~/git/'+ name + '.pkl'), 'wb') as f:
		pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


static_fac, lam = sys.argv[1:]
#iters = int(iters)
lam = float(lam)


import pickle
pred = {}
for random_seed in range(10):
	pred[random_seed] = load_obj('pred_normal_' + static_fac + '_' + str(lam) + '_' + str(random_seed) + '_const')

out = {}
for random_seed in range(10):
	out[random_seed] = {}
	for appliance in APPLIANCES_ORDER[1:]:
		out[random_seed][appliance] = {}
		for f in range(10, 110, 10):
			print random_seed, appliance, f
			s = pd.concat(pred[random_seed][appliance][f]).ix[sd_df.index]
			if appliance=="hvac":
				out[random_seed][appliance][f] = compute_rmse_fraction(appliance,s[range(4, 10)],'SanDiego')[2]
			else:   
				out[random_seed][appliance][f] = compute_rmse_fraction(appliance, s,'SanDiego')[2]

save_obj(out, "out_normal_" + static_fac + '_' + str(lam) + '_const')
