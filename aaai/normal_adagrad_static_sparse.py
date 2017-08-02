from sklearn.model_selection import KFold
from create_matrix import *
from tensor_custom_core import *
from create_matrix import *
from tensor_custom_core import *
from degree_days import dds
import os
from sklearn.model_selection import train_test_split, KFold
import sys
import pickle

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

appliance_index = {appliance: APPLIANCES_ORDER.index(appliance) for appliance in APPLIANCES_ORDER}

APPLIANCES = ['fridge', 'hvac', 'wm', 'mw', 'oven', 'dw']
region = "SanDiego"
year = 2014
cost='abs'

au_df, au_dfc = create_matrix_single_region("Austin", year)
au_tensor = get_tensor(au_df, au_dfc)
static_au = au_df[['area','total_occupants','num_rooms']].copy()
static_au['area'] = static_au['area'].div(4000)
static_au['total_occupants'] = static_au['total_occupants'].div(8)
static_au['num_rooms'] = static_au['num_rooms'].div(8)
static_au = static_au.values

df, dfc = create_matrix_single_region("SanDiego", year)
tensor = get_tensor(df, dfc)
static_sd = df[['area','total_occupants','num_rooms']].copy()
static_sd['area'] = static_sd['area'].div(4000)
static_sd['total_occupants'] = static_sd['total_occupants'].div(8)
static_sd['num_rooms'] = static_sd['num_rooms'].div(8)
static_sd = static_sd.values


static_fac, lam, random_seed = sys.argv[1:]
lam = float(lam)
random_seed = int(random_seed)

pred = {}
sd = {}
out = {}
n_splits = 10
n_iter = 3000
TRAIN_SPLITS = range(10, 110, 10)
case = 2
num_home = 5
a = 3

cost = 'l21'
algo = 'adagrad'

<<<<<<< HEAD
if static_fac is None:
    print ('here')
    H_known_Sd = None
=======
if static_fac == 'None':
	print 'None'
	H_known_Sd = None
>>>>>>> 0bc20a60ba19c03ee238aec36ca46544b32e13a3
else:
	print 'static'
	H_known_Sd = static_sd

for appliance in APPLIANCES_ORDER:
    pred[appliance] = {f:[] for f in range(10, 110, 10)}


kf = KFold(n_splits=n_splits)

for train_percentage in TRAIN_SPLITS:
#    print(lam, static_fac, random_seed, train_percentage)
    rd = 0
    for train_max, test in kf.split(df):
	print (static_fac, lam, random_seed, train_percentage, rd)
	rd += 1
	
        num_train = int((train_percentage*len(train_max)/100)+0.5)
        if train_percentage==100:
            train = train_max
        else:
            train, _ = train_test_split(train_max, train_size = train_percentage/100.0, random_state=random_seed)
        train_ix = df.index[train]
        test_ix = df.index[test]

        num_test = len(test_ix)
        train_test_ix = np.concatenate([test_ix, train_ix])
        df_t, dfc_t = df.ix[train_test_ix], dfc.ix[train_test_ix]
        tensor = get_tensor(df_t, dfc_t)
        tensor_copy = tensor.copy()
        # First n
        tensor_copy[:num_test, 1:, :] = np.NaN

        if static_fac == 'static':
            H, A, T, Hs, As, Ts, HATs, costs = learn_HAT_adagrad(case, tensor_copy, num_home, a, num_iter=n_iter, lr=1, dis=False, cost_function=cost, H_known=H_known_Sd[np.concatenate([test, train])], T_known = np.ones(12).reshape(-1, 1), penalty_coeff=lam)
        else:
            H, A, T, Hs, As, Ts, HATs, costs = learn_HAT_adagrad(case, tensor_copy, num_home, a, num_iter=n_iter, lr=1, dis=False, cost_function=cost, T_known = np.ones(12).reshape(-1, 1), penalty_coeff=lam)

        HAT = multiply_case(H, A, T, case)
        for appliance in APPLIANCES_ORDER:
            pred[appliance][train_percentage].append(pd.DataFrame(HAT[:num_test, appliance_index[appliance], :], index=test_ix))

save_obj(pred, "pred_normal_" + static_fac + "_" + str(lam) + "_" + str(random_seed) + "_const")


