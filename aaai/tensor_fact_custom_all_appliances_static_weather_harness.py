from create_matrix import *

from tensor_custom_core import *
from create_matrix import *
from tensor_custom_core import *
from degree_days import dds
appliance_index = {appliance: APPLIANCES_ORDER.index(appliance) for appliance in APPLIANCES_ORDER}

APPLIANCES = ['fridge', 'hvac', 'wm', 'mw', 'oven', 'dw']
region = "Austin"
year = 2014

import os
if region=="Austin":
	base_path = os.path.expanduser("~/scalable/filtered_tf_all_appliances_static_weather/")
else:
	base_path = os.path.expanduser("~/scalable/sd/tf_all_appliances_static_weather/")

def un_normalize(x, maximum, minimum):
	return (maximum - minimum) * x + minimum





import sys
a, static, weather = sys.argv[1:]
a = int(a)

case=2
cost='abs'

df, dfc = create_matrix_single_region(region, year)
start, stop = 1, 13
energy_cols = np.array(
	[['%s_%d' % (appliance, month) for month in range(start, stop)] for appliance in APPLIANCES_ORDER]).flatten()

static_cols = ['area', 'total_occupants', 'num_rooms']
static_df = df[static_cols]
static_df = static_df.div(static_df.max())
weather_values = np.array(dds[2014]['Austin'][start - 1:stop - 1]).reshape(-1, 1)

dfc = df.copy()

df = dfc[energy_cols]
col_max = df.max().max()
col_min = df.min().min()
# df = (1.0 * (df - col_min)) / (col_max - col_min)
tensor = df.values.reshape((len(df), 7, stop - start))
M, N, O = tensor.shape

if static=="static":
	H_known = static_df.values[:, :a]
else:
	H_known = None

if weather=='weather':
	T_known = weather_values
else:
	T_known=None

pred = {}
# Find location of test home in the tensor
for i in range(len(df))[:]:
	home_num = df.index.values[i]
	print home_num, i
	tensor_copy = tensor.copy()
	tensor_copy[i, 1:, :] = np.NaN
	# H, A, T = learn_HAT(case, tensor_copy, a, a, num_iter=2000, lr=0.1, dis=True, cost_function=cost, H_known=static_df.values[:, :1],
	#                   T_known=weather_values)
	H, A, T = learn_HAT(case, tensor_copy, a, a, num_iter=2000, lr=0.1, dis=False, cost_function=cost, H_known=H_known, T_known=T_known)
	HAT = multiply_case(H, A, T, case)
	for appliance in APPLIANCES_ORDER:
		if appliance not in pred:
			pred[appliance] = {}
		pred[appliance][home_num] = HAT[i, appliance_index[appliance], :]


for appliance in APPLIANCES_ORDER:
	store_path = "/".join([str(x) for x in [appliance, static, weather, a]])
	store_path = os.path.join(base_path, store_path)
	if not os.path.exists(store_path):
		os.makedirs(store_path)
	store_path = store_path + "/data.csv"
	out_df = pd.DataFrame(pred[appliance]).T
	out_df.to_csv(store_path)
