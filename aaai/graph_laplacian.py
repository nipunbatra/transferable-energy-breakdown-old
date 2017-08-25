from sklearn.cluster import DBSCAN
from sklearn.model_selection import KFold
import pandas as pd
from create_matrix import *
from tensor_custom_core import *
from create_matrix import *
from tensor_custom_core import *
from degree_days import dds
import os
from degree_days import dds
import autograd.numpy as np
import pickle
from sklearn.model_selection import train_test_split, KFold
from scipy.spatial.distance import pdist, squareform




appliance_index = {appliance: APPLIANCES_ORDER.index(appliance) for appliance in APPLIANCES_ORDER}

APPLIANCES = ['fridge', 'hvac', 'wm', 'mw', 'oven', 'dw']
region = "SanDiego"
year = 2014


def un_normalize(x, maximum, minimum):
    return (maximum - minimum) * x + minimum

def get_tensor(df):
    start, stop = 1, 13
    energy_cols = np.array(
        [['%s_%d' % (appliance, month) for month in range(start, stop)] for appliance in APPLIANCES_ORDER]).flatten()

    dfc = df.copy()

    df = dfc[energy_cols]

    tensor = df.values.reshape((len(df), 7, stop - start))
    return tensor

def create_region_df_dfc_static(region, year):
    df, dfc = create_matrix_single_region(region, year)
    tensor = get_tensor(df)
    static_region = df[['area', 'total_occupants', 'num_rooms']].copy()
    static_region['area'] = static_region['area'].div(4000)
    static_region['total_occupants'] = static_region['total_occupants'].div(8)
    static_region['num_rooms'] = static_region['num_rooms'].div(8)
    static_region =static_region.values
    return df, dfc, tensor, static_region

def distance(x, y):
    return np.linalg.norm(x - y)

def fill_missing(X):
    n_sample, n_feature = X.shape
    mean = np.mean(X, axis=0)
    for i in range(n_feature):
        X.iloc[:, i].loc[X.iloc[:, i].isnull()] = mean[i]
    return X

from sklearn.neighbors import NearestNeighbors

def get_L_NN(X):
    nbrs = NearestNeighbors(n_neighbors=5, radius = 0.05, algorithm='ball_tree').fit(X)
    distances, indices = nbrs.kneighbors(X)
    n_sample, n_feature = X.shape
    W = np.zeros((n_sample, n_sample))
    for i in range(n_sample):
        if distances[i][4] == 0:
            continue
        for j in indices[i]:
            W[i][j] = 1
            W[j][i] = 1
    K = np.dot(W, np.ones((n_sample, n_sample)))
    D = np.diag(np.diag(K))
    return D - W

def get_L(X):
    W = 1-squareform(pdist(X, 'cosine'))
    W = np.nan_to_num(W)
    n_sample, n_feature = W.shape
    
    K = np.dot(W, np.ones((n_sample, n_sample)))
    D = np.diag(np.diag(K))
    return D - W

def cost_graph_laplacian(H, A, T, L, E_np_masked, lam, case):
    HAT = multiply_case(H, A, T, case)
    mask = ~np.isnan(E_np_masked)
    error_1 = (HAT - E_np_masked)[mask].flatten()
    
    HTL = np.dot(H.T, L)
    HTLH = np.dot(HTL, H)
    error_2 = np.trace(HTLH)

    return np.sqrt((error_1**2).mean()) + lam * error_2

def learn_HAT_adagrad_graph(case, E_np_masked, L, a, b, num_iter=2000, lr=0.1, dis=False, lam = 1, H_known=None,A_known=None, T_known=None, random_seed=0, eps=1e-8, penalty_coeff=0.0):

    np.random.seed(random_seed)
    cost = cost_graph_laplacian
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
    
    H = np.random.rand(*H_dim)
    A = np.random.rand(*A_dim)
    T = np.random.rand(*T_dim)

    sum_square_gradients_H = np.zeros_like(H)
    sum_square_gradients_A = np.zeros_like(A)
    sum_square_gradients_T = np.zeros_like(T)

    Hs = [H.copy()]
    As = [A.copy()]
    Ts = [T.copy()]
    
    costs = [cost(H, A, T, L, E_np_masked, lam, case)]
    
    HATs = [multiply_case(H, A, T, case)]

    # GD procedure
    for i in range(num_iter):
        del_h, del_a, del_t = mg(H, A, T, L, E_np_masked, lam, case)
        sum_square_gradients_H += eps + np.square(del_h)
        sum_square_gradients_A += eps + np.square(del_a)
        sum_square_gradients_T += eps + np.square(del_t)

        

        lr_h = np.divide(lr, np.sqrt(sum_square_gradients_H))
        lr_a = np.divide(lr, np.sqrt(sum_square_gradients_A))
        lr_t = np.divide(lr, np.sqrt(sum_square_gradients_T))

        H -= lr_h * del_h
        A -= lr_a * del_a
        T -= lr_t * del_t
        # Projection to known values
        if H_known is not None:
            H = set_known(H, H_known)
        if A_known is not None:
            A = set_known(A, A_known)
        if T_known is not None:
            T = set_known(T, T_known)
        # Projection to non-negative space
        H[H < 0] = 1e-8
        A[A < 0] = 1e-8
        T[T < 0] = 1e-8

        As.append(A.copy())
        Ts.append(T.copy())
        Hs.append(H.copy())
        
        costs.append(cost(H, A, T, L, E_np_masked, lam, case))

        HATs.append(multiply_case(H, A, T, case))
        if i % 500 == 0:
            if dis:
                print(cost(H, A, T, L, E_np_masked, lam, case))
    return H, A, T, Hs, As, Ts, HATs, costs

au_df, au_dfc, au_tensor, au_static = create_region_df_dfc_static('Austin', year)
sd_df, sd_dfc, sd_tensor, sd_static = create_region_df_dfc_static('SanDiego', year)

# using aggregate reading with KNN 
au_agg = au_df.loc[:, 'aggregate_1':'aggregate_12'].copy()
sd_agg = sd_df.loc[:, 'aggregate_1':'aggregate_12'].copy()

au_agg = np.nan_to_num(au_agg)
sd_agg = np.nan_to_num(sd_agg)

L_au = get_L_NN(au_agg)
L_sd = get_L_NN(sd_agg)


# using KNN to compute L
# au_static_copy = au_static.copy()
# sd_static_copy = sd_static.copy()
# au_static_copy = np.nan_to_num(au_static_copy)
# sd_static_copy = np.nan_to_num(sd_static_copy)
# L_au = get_L_NN(au_static_copy)
# L_sd = get_L_NN(sd_static_copy)

# # using cosine similarity to compute L
# L_au = get_L(au_static)
# L_sd = get_L(sd_static)

lam= sys.argv[1]
lam = float(lam)

n_splits = 10
case = 2
a = 5
b = 3
c = 3
iters = 2000


H_au, A_au, T_au, Hs, As, Ts, HATs, costs = learn_HAT_adagrad_graph(case, au_tensor, L_au, a, b, num_iter=2000, lr=0.1, dis=False, lam=lam, T_known = np.ones(12).reshape(-1, 1))


pred_normal = {}
pred_transfer = {}
for random_seed in range(5):
    pred_normal[random_seed] = {}
    pred_transfer[random_seed] = {}
    
    for appliance in APPLIANCES_ORDER:
        pred_normal[random_seed][appliance] = {f:[] for f in range(10, 110, 20)}
        pred_transfer[random_seed][appliance] = {f:[] for f in range(10, 110, 20)}

kf = KFold(n_splits=n_splits)
for random_seed in range(5):
    print "random seed: ", random_seed
    np.random.seed(random_seed)
    # H_au, A_au, T_au, Hs, As, Ts, HATs, costs = learn_HAT_adagrad_graph(case, au_tensor, L_au, a, b, num_iter=2000, lr=0.1, dis=False, lam=lam, T_known = np.ones(12).reshape(-1, 1), random_seed = random_seed)
    for train_percentage in range(10, 110, 20):
        print "training percentage: ", train_percentage
        rd = 0
        
        for train_max, test in kf.split(sd_df):
            print "round: ", rd
           
            
            num_train = int((train_percentage*len(train_max)/100)+0.5)
            num_test = len(test)
            
            # get the random training data from train_max based on then random seed
            if train_percentage==100:
                train = train_max
            else:
                train, _ = train_test_split(train_max, train_size = train_percentage/100.0)
            
            # get the index of training and testing data
            train_ix = sd_df.index[train]
            test_ix = sd_df.index[test]
            print "test_ix: ", test_ix
            
            # create the tensor
            train_test_ix = np.concatenate([test_ix, train_ix])
            df_t, dfc_t = sd_df.ix[train_test_ix], sd_dfc.ix[train_test_ix]
            tensor = get_tensor(df_t)

            
            ############################################################################################
            # Normal learning: no constant constraint, no A_known, with learn_HAT
            ############################################################################################
            tensor_copy = tensor.copy()
            tensor_copy[:num_test, 1:, :] = np.NaN
            # agg = sd_agg[np.concatenate([test_ix, train_ix])]
            L = L_sd[np.ix_(np.concatenate([test, train]), np.concatenate([test, train]))]

#             H, A, T, F = learn_HAT_graph(2, tensor_copy, static_sd[np.concatenate([test, train])], sim_sd, a, b, num_iter=iters,dis=False, T_known = np.ones(12).reshape(-1, 1))
            H, A, T, Hs, As, Ts, HATs, costs = learn_HAT_adagrad_graph(case, tensor_copy, L, a, b, num_iter=2000, lr=0.1, dis=False, lam=lam, T_known = np.ones(12).reshape(-1, 1))
            
            # get the prediction
            HAT = multiply_case(H, A, T, case)
            for appliance in APPLIANCES_ORDER:
                pred_normal[random_seed][appliance][train_percentage].append(pd.DataFrame(HAT[:num_test, appliance_index[appliance], :], index=test_ix))     

            ############################################################################################
            # transfer learning: constant constraint, with A_known = A_a_const, with learn_HAT_constant
            ############################################################################################
            tensor_copy = tensor.copy()
            tensor_copy[:num_test, 1:, :] = np.NaN
            # agg = sd_agg[np.concatenate([test, train])]
            L = L_sd[np.ix_(np.concatenate([test, train]), np.concatenate([test, train]))]
            
#             H, A, T, F = learn_HAT_graph(2, tensor_copy, static_sd[np.concatenate([test, train])], sim_sd, a, b, num_iter=iters,dis=False, T_known = np.ones(12).reshape(-1, 1))
#             H, A, T, F, Hs, As, Ts, Fs, HATs, costs = learn_HAT_adagrad_graph(case, tensor_copy, L, agg, a, b, num_iter=20000, lr=0.1, dis=False,A_known = A_au)
            H, A, T, Hs, As, Ts, HATs, costs = learn_HAT_adagrad_graph(case, tensor_copy, L, a, b, num_iter=2000, lr=0.1, dis=False, lam=lam, A_known = A_au, T_known = np.ones(12).reshape(-1, 1))

      
            # get the prediction
            HAT = multiply_case(H, A, T, case)
            for appliance in APPLIANCES_ORDER:
                pred_transfer[random_seed][appliance][train_percentage].append(pd.DataFrame(HAT[:num_test, appliance_index[appliance], :], index=test_ix))
            
            
            rd += 1


def save_obj(obj, name):
    with open(os.path.expanduser('~/git/graph_test_2/'+ name + '.pkl'), 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

save_obj(pred_normal, "normal_{}".format(lam))
save_obj(pred_transfer, "transfer_{}".format(lam))






