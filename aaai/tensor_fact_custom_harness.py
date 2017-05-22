from create_matrix import *

from tensor_custom_core import *

APPLIANCES = ['fridge', 'hvac', 'wm', 'mw', 'oven', 'dw']
region = "Austin"
year = 2014

import os
base_path = os.path.expanduser("~/scalable/tf/")

def un_normalize(x, maximum, minimum):
	return (maximum - minimum) * x + minimum




def tf_factorise(appliance, case, a, cost, all_features):
	if appliance == "hvac":
		start, stop = 5, 11
	else:
		start, stop = 1, 13
	months = stop - start
	appliance_df = create_matrix_region_appliance_year(region, year, appliance, all_features)
	aggregate_cols = [x for x in appliance_df.columns if "aggregate" in x]
	appliance_cols = [x for x in appliance_df.columns if appliance in x]
	energy_cols = np.concatenate([aggregate_cols, appliance_cols])

	df = appliance_df.copy()
	dfc = df.copy()

	df = df[energy_cols]
	col_max = df.max().max()
	col_min = df.min().min()
	df = (1.0 * (df - col_min)) / (col_max - col_min)
	tensor = df.values.reshape((len(df), 2, months))
	M, N, O = tensor.shape
	mask = np.ones(M).astype('bool')
	pred = {}
	for i, home in enumerate(df.index[:]):
		try:
			tensor_copy = tensor.copy()
			tensor_copy[i, 1, :] = np.NaN

			H, A, T = learn_HAT(case, tensor_copy, a, a, num_iter=2000, lr=0.1, dis=False, cost_function=cost)
			prediction = multiply_case(H, A, T, case)
			pred_appliance = un_normalize(prediction[i, 1, :], col_max, col_min)
			pred[home] = pred_appliance
			print i, home, pred_appliance, un_normalize(tensor[i, 1, :], col_max, col_min)
		except:
			pass
	pred = pd.DataFrame(pred).T
	return pred


import sys
appliance, case, a, cost, all_features = sys.argv[1:]
case = int(case)
a = int(a)
if all_features =='True':
	all_features =True
else:
	all_features = False

print appliance, case, a, cost, all_features


out = tf_factorise(appliance, case, a, cost, all_features)

store_path = "/".join([str(x) for x in [appliance, case, a, cost, all_features]])
store_path = os.path.join(base_path, store_path)
if not os.path.exists(store_path):
	os.makedirs(store_path)
store_path = store_path + "/data.csv"

out.to_csv(store_path)
