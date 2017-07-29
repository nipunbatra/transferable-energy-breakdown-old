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

def load_obj(name ):
    with open(os.path.expanduser('~/git/' + name + '.pkl'), 'rb') as f:
        return pickle.load(f)


iter_train, random_seed= sys.argv[1:]
iter_train = int(iter_train)
random_seed = int(random_seed)

print iter_train, random_seed

appliance_index = {appliance: APPLIANCES_ORDER.index(appliance) for appliance in APPLIANCES_ORDER}
APPLIANCES = ['fridge', 'hvac', 'wm', 'mw', 'oven', 'dw']
year = 2014

region = "SanDiego"
sd_df, sd_dfc = create_matrix_single_region("SanDiego", year)
sd_tensor = get_tensor(sd_df, sd_dfc)
region = "Austin"
au_df, au_dfc = create_matrix_single_region("Austin", year)
au_tensor = get_tensor(au_df, au_dfc)

pred = {}
n_splits = 10
case = 2
iter_adapt = 3000
a = 3
cost = 'abs'


H_au, A_au, T_au = learn_HAT_adagrad(case, au_tensor, a, a, num_iter=iter_train, lr=0.1, dis=False, cost_function=cost, T_known=np.ones(12).reshape(-1, 1))



for appliance in APPLIANCES_ORDER:
    pred[appliance] = {f:[] for f in range(10, 110, 10)}

# print method, algo, cost, a, lr, random_seed
print "pred_" + str(iter_train)

kf = KFold(n_splits=n_splits)
for train_percentage in range(10, 110, 10):
    print "tran percentage: ", train_percentage
    rd = 0
    for train_max, test in kf.split(sd_df):
        print "round: ", rd
        rd += 1

        num_train = int((train_percentage*len(train_max)/100)+0.5)
        if train_percentage==100:
            train = train_max
        else:
            train, _ = train_test_split(train_max, train_size = train_percentage/100.0, random_state=random_seed)
        train_ix = sd_df.index[train]
        test_ix = sd_df.index[test]

        num_test = len(test_ix)
        train_test_ix = np.concatenate([test_ix, train_ix])
        df_t, dfc_t = sd_df.ix[train_test_ix], sd_dfc.ix[train_test_ix]
        tensor = get_tensor(df_t, dfc_t)
        tensor_copy = tensor.copy()
        # First n
        tensor_copy[:num_test, 1:, :] = np.NaN
        
        H, A, T = learn_HAT_adagrad(case, tensor_copy, a, a, num_iter=iter_adapt, lr=0.1, dis=False, cost_function=cost, A_known=A_au, T_known=np.ones(12).reshape(-1, 1))

        HAT = multiply_case(H, A, T, case)
        for appliance in APPLIANCES_ORDER:
            pred[appliance][train_percentage].append(pd.DataFrame(HAT[:num_test, appliance_index[appliance], :], index=test_ix))

save_obj(pred, "pred_" + str(iter_train) + "_" + str(random_seed))
