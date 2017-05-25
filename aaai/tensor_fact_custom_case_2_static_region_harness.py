import os
from create_matrix import *
from degree_days import dds
from tensor_custom_core import *



def un_normalize(x, maximum, minimum):
    return (maximum-minimum)*x + minimum

region="Austin"
year=2014
appliance, ALL_FEATURES, a, h, t, cost = sys.argv[1:]
a = int(a)
if ALL_FEATURES=="True":
	ALL_FEATURES=True
else:
	False

if appliance == "hvac":
	start, stop = 5, 11
else:
	start, stop = 1, 13
months = stop - start
appliance_df = create_matrix_region_appliance_year(region, year, appliance, all_features=ALL_FEATURES)
aggregate_cols = [x for x in appliance_df.columns if "aggregate" in x]
appliance_cols = [x for x in appliance_df.columns if appliance in x]
energy_cols = np.concatenate([aggregate_cols, appliance_cols])

df = appliance_df.copy()
dfc = df.copy()
static_cols = ['area', 'total_occupants', 'num_rooms']
static_df = df[static_cols]
static_df = static_df.div(static_df.max())
weather_values = np.array(dds[2014]['Austin'][start-1:stop-1]).reshape(-1,1)

df = df[energy_cols]
col_max = df.max().max()
col_min = df.min().min()
df = (1.0*(df-col_min))/(col_max-col_min)
tensor = df.values.reshape((len(df), 2, months))
M, N, O = tensor.shape
mask = np.ones(M).astype('bool')

case=2
if h == "static":
	H_known = static_df.values[:,:a]
else:
	H_known = None

if t=="weather":
	T_known = weather_values
else:
	T_known = None

pred = {}

for i, home in enumerate(df.index[:]):
	try:


		tensor_copy = tensor.copy()
		tensor_copy[i, 1, :]=np.NaN
		H, A, T = learn_HAT(case, tensor_copy, a, a, num_iter=2000, lr=0.1, dis=False, cost_function=cost,
		                    H_known=H_known, T_known=T_known)
		prediction = multiply_case(H, A, T, case)

		pred_appliance = un_normalize(prediction[i, 1, :], col_max, col_min)
		pred[home] = pred_appliance
	except:
		pass

out_df = pd.DataFrame(pred).T

base_path = os.path.expanduser("~/scalable/tf_energy_static/")
if not os.path.exists(base_path):
	os.makedirs(base_path)
out = pd.DataFrame(pred).T
store_path = "/".join([str(x) for x in [appliance, ALL_FEATURES, a, h, t, cost]])
store_path = os.path.join(base_path, store_path)
if not os.path.exists(store_path):
	os.makedirs(store_path)
store_path = store_path + "/data.csv"

out.to_csv(store_path)
