import os

from aaai18.common import compute_rmse_fraction
from create_matrix import *
from degree_days import dds
from tensor_custom_core import *

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
	with open(os.path.expanduser('~/git/pred_explore/' + name + '.pkl'), 'rb') as f:
		return pickle.load(f)
def save_obj(obj, name ):
	with open(os.path.expanduser('~/git/out_explore/'+ name + '.pkl'), 'wb') as f:
		pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


iters, algo, static_fac, lam = sys.argv[1:]
iters = int(iters)
lam = float(lam)


import pickle
pred = {}
pred['normal'] = {}
pred['transfer'] = {}
for random_seed in range(10):
	pred['transfer'][random_seed] = load_obj('pred_transfer_' + str(iters) + '_' + algo + '_' + static_fac + '_' + str(lam) + '_' + str(random_seed))
	pred['normal'][random_seed] = load_obj('pred_normal_' + str(iters) + '_' + algo + '_' + static_fac + '_' + str(lam) + '_' + str(random_seed))

out = {}
out['normal'] = {}
out['transfer'] = {}
for random_seed in range(10):
	out['normal'][random_seed] = {}
	out['transfer'][random_seed] = {}
	for appliance in APPLIANCES_ORDER[1:]:
		out['normal'][random_seed][appliance] = {}
		out['transfer'][random_seed][appliance] = {}
		for f in range(10, 110, 10):
			print random_seed, appliance, f
			s_transfer = pd.concat(pred['transfer'][random_seed][appliance][f]).ix[sd_df.index]
			s_normal = pd.concat(pred['normal'][random_seed][appliance][f]).ix[sd_df.index]
			if appliance=="hvac":
				out['transfer'][random_seed][appliance][f] = compute_rmse_fraction(appliance,s_transfer[range(4, 10)],'SanDiego')[2]
				out['normal'][random_seed][appliance][f] = compute_rmse_fraction(appliance,s_normal[range(4, 10)],'SanDiego')[2]
			else:   
				out['transfer'][random_seed][appliance][f] = compute_rmse_fraction(appliance,s_transfer,'SanDiego')[2]
				out['normal'][random_seed][appliance][f] = compute_rmse_fraction(appliance,s_normal,'SanDiego')[2]

save_obj(out['transfer'], "out_transfer_" + str(iters) + '_' + algo + '_' + static_fac + '_' + str(lam))
save_obj(out['normal'], "out_normal_" + str(iters) + '_' + algo + '_' + static_fac + '_' + str(lam))
