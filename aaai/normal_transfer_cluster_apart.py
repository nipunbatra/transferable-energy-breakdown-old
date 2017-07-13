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


method, cost, iterations = sys.argv[1:]
iterations = int(iterations)

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
a = 2
b = 3
num_seed = 10
# iters = [2000, 10000]
#cost = 'abs'

if method == "transfer":
	H_a, A_a, T_a = learn_HAT(case, au_tensor, a, b, num_iter = iterations, lr = 0.1, dis = False,
				cost_functions = cost, A_known = A_a, T_known = np.ones(12).reshape(-1, 1))

for random_seed in range(num_seed):
    pred[random_seed] = {}
    for appliance in APPLIANCES_ORDER:
        pred[random_seed][appliance] = {f:[] for f in range(10, 110, 10)}


kf = KFold(n_splits = n_splits)
for random_seed in range(num_seed):
    print "random seed: ", random_seed

    for train_percentage in range(10, 110, 10):
        print "training percentage: ", train_percentage
        rd = 0

        for train_max, test in kf.split(sd_df):
            print "round: ", rd
            rd += 1

            num_train = int((train_percentage*len(train_max)/100)+0.5)
            num_test = len(test)

            # get the random training data from train_max based on then random seed
            if train_percentage==100:
                train = train_max
            else:
                train, _ = train_test_split(train_max, train_size = train_percentage/100.0, random_state=random_seed)

            # get the index of training and testing data
            train_ix = sd_df.index[train]
            test_ix = sd_df.index[test]

            # create the tensor
            train_test_ix = np.concatenate([test_ix, train_ix])
            df_t, dfc_t = sd_df.ix[train_test_ix], sd_dfc.ix[train_test_ix]
            tensor = get_tensor(df_t, dfc_t)
            tensor_copy = tensor.copy()
            # set the appliance consumption to be missing for testing data
            tensor_copy[:num_test, 1:, :] = np.NaN
            
            if method == "transfer":
                # transfer learning
#                H_a, A_a, T_a = learn_HAT(case, au_tensor, a, b, num_iter=iterations, lr=0.1, dis=False, 
#                                         cost_function=cost, T_known=np.ones(12).reshape(-1, 1))
                H, A, T = learn_HAT(case, tensor_copy, a, b, num_iter=iterations, lr=0.1, dis=False, 
                                    cost_function=cost, A_known = A_a, T_known=np.ones(12).reshape(-1, 1))
            else:
                # normal learning
                H, A, T = learn_HAT(case, tensor_copy, a, b, num_iter=iterations, lr=0.1, dis=False, cost_function=cost, T_known=np.ones(12).reshape(-1, 1))

            # get the prediction
            HAT = multiply_case(H, A, T, case)
            for appliance in APPLIANCES_ORDER:
                pred[random_seed][appliance][train_percentage].append(pd.DataFrame(HAT[:num_test, appliance_index[appliance], :], index=test_ix))

# pred.to_pickle("~/git/pred.pkl") 
save_obj(pred, "pred_" + method + "_" + cost + "_" + str(iterations))         


