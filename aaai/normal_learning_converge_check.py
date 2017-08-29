import os
import pickle

from create_matrix import *
from degree_days import dds
from tensor_custom_core import *


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
def learn_HAT_random_normal_check(case, E_np_masked, a, b, num_iter=2000, lr=0.1, dis=False, cost_function='abs', H_known=None,
          A_known=None, T_known=None, random_seed=0, random_mul_constant=1,
              random_add_constant=0):
    np.random.seed(random_seed)
    if cost_function == 'abs':
        cost = cost_abs
    else:
        cost = cost_rel
    mg = multigrad(cost, argnums=[0, 1, 2])

    params = {}
    params['M'], params['N'], params['O'] = E_np_masked.shape
    params['a'] = a
    params['b'] = b
    H_dim_chars = list(cases[case]['HA'].split(",")[0].strip())
    H_dim = tuple(params[x] for x in H_dim_chars)
    A_dim_chars = list(cases[case]['HA'].split(",")[1].split("-")[0].strip())
    A_dim = tuple(params[x] for x in A_dim_chars)
    T_dim_chars = list(cases[case]['HAT'].split(",")[1].split("-")[0].strip())
    T_dim = tuple(params[x] for x in T_dim_chars)
    H = np.random.rand(*H_dim)*random_mul_constant+random_add_constant

    A = np.random.rand(*A_dim)*random_mul_constant+random_add_constant
    T = np.random.rand(*T_dim)*random_mul_constant+random_add_constant

    Hs =[H.copy()]
    As= [A.copy()]
    Ts = [T.copy()]
    costs = [cost_abs(H, A, T, E_np_masked, 2)]
    HATs =[multiply_case(H, A, T, 2)]
    # GD procedure
    for i in range(num_iter):
        del_h, del_a, del_t = mg(H, A, T, E_np_masked, case)
        H -= lr * del_h
        A -= lr * del_a
        T -= lr * del_t
        # Projection to known values
        if H_known is not None:
            H = set_known(H, H_known)
        if A_known is not None:
            A = set_known(A, A_known)
        if T_known is not None:
            T = set_known(T, T_known)
        # Projection to non-negative space
        H[H < 0] = 0
        A[A < 0] = 0
        T[T < 0] = 0
        As.append(A.copy())
        Ts.append(T.copy())
        Hs.append(H.copy())
        costs.append(cost_abs(H, A, T, E_np_masked, 2))
        HATs.append(multiply_case(H, A, T, 2))
        if i % 500 == 0:
            if dis:
                print(cost(H, A, T, E_np_masked, case))
    return H, A, T, Hs, As, Ts, HATs, costs


re, iterations = sys.argv[1:]
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

if re == "Austin":
	tensor = au_tensor.copy()
if re == "SanDiego":
	tensor = sd_tensor.copy()

out = {}
for scale in [1]:
    out[scale] = {}
    for random_seed in range(1):    
        out[scale][random_seed] = {}
		for c in [1]:
		    out[scale][random_seed][c] = {}
		    print (scale, random_seed, c)
		    tensor_copy = tensor.copy()
		    ################# PLEASE SEE THE LINE BELOW- CHANGED FROM learn_HAT to learn_HAT_random_normal #######
		    H, A, T, Hs, As, Ts, HATs, costs = learn_HAT_random_normal_check(case, tensor, a, 3, num_iter=iterations, lr=0.1, dis=False, cost_function=cost,T_known=np.ones(12).reshape(-1, 1))
		    out[scale][random_seed][c] = {'Hs':Hs, 'As':As, 'Ts':Ts, 'HATs':HATs, 'costs':costs}
# pred.to_pickle("~/git/pred.pkl") 
			save_obj(out, "out_" + re + "_normal_1_" + str(iterations))         


