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
    with open(os.path.expanduser('~/git/pred_static/'+ name + '.pkl'), 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

appliance_index = {appliance: APPLIANCES_ORDER.index(appliance) for appliance in APPLIANCES_ORDER}

APPLIANCES = ['fridge', 'hvac', 'wm', 'mw', 'oven', 'dw']
region = "SanDiego"
year = 2014

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

method, algo, static_fac, random_seed = sys.argv[1:]
random_seed = int(random_seed)

pred = {}

n_splits = 10
n_iter = 3000
TRAIN_SPLITS = range(10, 110, 10)
case = 2


kf = KFold(n_splits=n_splits)


if static_fac == 'None':
    H_known_Au = None
    H_known_Sd = None
else:
    H_known_Au = static_au
    H_known_Sd = static_sd

for appliance in APPLIANCES_ORDER:
    pred[appliance] = {f:[] for f in range(10, 110, 10)}

a = 5
b = 3

# Learning A_au with static factors, depends on different algorithms
if method == 'transfer':
    if algo == 'adagrad':
        cost = 'l21'
        H_au, A_au, T_au, Hs, As, Ts, HATs, costs = learn_HAT_adagrad(case, au_tensor, a, b, num_iter=train_iter, lr=0.1, dis=False, cost_function=cost, H_known = H_known_Au, T_known=np.ones(12).reshape(-1, 1))
    else:
        cost = 'abs'
        H_au, A_au, T_au = learn_HAT(case, au_tensor, a, b, num_iter=train_iter, lr=0.1, dis=False, cost_function=cost, H_known = H_known_Au, T_known=np.ones(12).reshape(-1, 1))
    

np.random.seed(random_seed)
for train_percentage in TRAIN_SPLITS:
    rd = 0
    for train_max, test in kf.split(df):
        print (method, algo, static_fac, random_seed, train_percentage, rd)
        rd += 1

        num_train = int((train_percentage*len(train_max)/100)+0.5)
        if train_percentage==100:
            train = train_max
        else:
            train, _ = train_test_split(train_max, train_size = train_percentage/100.0)
        train_ix = df.index[train]
        test_ix = df.index[test]

        num_test = len(test_ix)
        train_test_ix = np.concatenate([test_ix, train_ix])
        df_t, dfc_t = df.ix[train_test_ix], dfc.ix[train_test_ix]
        tensor = get_tensor(df_t, dfc_t)
        

        # First n
        ################################################################
        # Normal learning in SanDiego
        ################################################################
        tensor_copy = tensor.copy()
        tensor_copy[:num_test, 1:, :] = np.NaN

        if method  == 'normal':
            if algo == 'adagrad':
                cost = 'l21'
                if static_fac == 'static':
                    H_normal, A_normal, T_normal, Hs, As, Ts, HATs, costs = learn_HAT_adagrad(case, tensor_copy, a, b, num_iter=n_iter, lr=0.1, dis=False, cost_function=cost, H_known = H_known_Sd[np.concatenate([test, train])], T_known = np.ones(12).reshape(-1, 1))
                else:
                    H_normal, A_normal, T_normal, Hs, As, Ts, HATs, costs = learn_HAT_adagrad(case, tensor_copy, a, b, num_iter=n_iter, lr=0.1, dis=False, cost_function=cost, T_known = np.ones(12).reshape(-1, 1))
            else:
                print 'here'
                cost = 'abs'
                if static_fac == 'static':
                    H_normal, A_normal, T_normal = learn_HAT(case, tensor_copy, a, b, num_iter=n_iter, lr=0.1, dis=False, cost_function=cost, H_known = H_known_Sd[np.concatenate([test, train])], T_known = np.ones(12).reshape(-1, 1))
                else:
                    H_normal, A_normal, T_normal = learn_HAT(case, tensor_copy, a, b, num_iter=n_iter, lr=0.1, dis=False, cost_function=cost, T_known = np.ones(12).reshape(-1, 1))
            HAT = multiply_case(H_normal, A_normal, T_normal, case)
        else:
            if algo == 'adagrad':
                cost = 'l21'
                if static_fac == 'static':
                    H_transfer, A_transfer, T_transfer, Hs, As, Ts, HATs, costs = learn_HAT_adagrad(case, tensor_copy, a, b, num_iter=n_iter, lr=0.1, dis=False, cost_function=cost, H_known = H_known_Sd[np.concatenate([test, train])], A_known = A_au, T_known = np.ones(12).reshape(-1, 1))
                else:
                    H_transfer, A_transfer, T_transfer, Hs, As, Ts, HATs, costs = learn_HAT_adagrad(case, tensor_copy, a, b, num_iter=n_iter, lr=0.1, dis=False, cost_function=cost, A_known = A_au, T_known = np.ones(12).reshape(-1, 1), penalty_coeff=lam)
            else:
                cost = 'abs'
                if static_fac =='static':
                    H_transfer, A_transfer, T_transfer = learn_HAT(case, tensor_copy, a, b, num_iter=n_iter, lr=0.1, dis=False, cost_function=cost, H_known = H_known_Sd[np.concatenate([test, train])], A_known = A_au, T_known=np.ones(12).reshape(-1, 1))
                else:
                    H_transfer, A_transfer, T_transfer = learn_HAT(case, tensor_copy, a, b, num_iter=n_iter, lr=0.1, dis=False, cost_function=cost, A_known = A_au, T_known=np.ones(12).reshape(-1, 1))
            HAT = multiply_case(H_transfer, A_transfer, T_transfer, case)
       

        for appliance in APPLIANCES_ORDER:
            pred[appliance][train_percentage].append(pd.DataFrame(HAT[:num_test, appliance_index[appliance], :], index=test_ix))
name = "pred_{}_{}_{}_{}".format(method, algo, static_fac, random_seed)
save_obj(pred, name)

