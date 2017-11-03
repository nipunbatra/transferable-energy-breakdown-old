
import datetime
from sklearn.model_selection import train_test_split, KFold
import sys
sys.path.insert(0, '../../aaai18/code/')
from tensor_custom_core import *
from common import *
from create_matrix import *

appliance_index = {appliance: APPLIANCES_ORDER.index(appliance) for appliance in APPLIANCES_ORDER}

APPLIANCES = ['fridge', 'hvac', 'wm', 'mw', 'oven', 'dw']
region = "Austin"
year = 2014

import os

def un_normalize(x, maximum, minimum):
    return (maximum - minimum) * x + minimum


# In[113]:

import sys
from degree_days import dds
case = 2
a = 3
b = 3
source = 'Austin'
target = 'SanDiego'
constant_use = 'True'
start = 1
stop = 13


# In[4]:

source_df, source_dfc, source_tensor, source_static = create_region_df_dfc_static(source, year, start, stop)
target_df, target_dfc, target_tensor, target_static = create_region_df_dfc_static(target, year, start, stop)

# # using cosine similarity to compute L
source_L = get_L(source_static)
target_L = get_L(target_static)

# Seasonal constant constraints
if constant_use == 'True':
    T_constant = np.ones(stop-start).reshape(-1 , 1)
else:
    T_constant = None


# In[5]:

def get_tensor_appliance(df, dfc, appliance):
    start, stop = 1, 13
    energy_cols = np.array(
        [['%s_%d' % (appliance, month) for month in range(start, stop)] ]).flatten()
    static_cols = ['area', 'total_occupants', 'num_rooms']
    static_df = df[static_cols]
    static_df = static_df.div(static_df.max())
    weather_values = np.array(dds[2014][region][start - 1:stop - 1]).reshape(-1, 1)

    dfc = df.copy()

    df = dfc[energy_cols]
    col_max = df.max().max()
    col_min = df.min().min()
    # df = (1.0 * (df - col_min)) / (col_max - col_min)
    tensor = df.values.reshape((len(df), 1, stop - start))
    M, N, O = tensor.shape
    return tensor


# In[6]:

agg_target = get_tensor_appliance(target_df, target_dfc, 'aggregate')
agg_source = get_tensor_appliance(source_df, source_dfc, 'aggregate')
agg_tensor = np.concatenate((agg_source, agg_target), axis=0)


# In[7]:

all_tensor = np.concatenate((source_tensor, target_tensor), axis=0)


# In[8]:

static = np.r_[source_static, target_static]
L = get_L(static)


# # Use aggregate readings to learn Home factors

# In[58]:

tensor_copy = agg_tensor.copy()
H_agg, A_agg, T_agg, Hs, As, Ts, HATs, costs = learn_HAT_adagrad_graph(case, tensor_copy,
                                                            L,
                                                          a,
                                                          b,
                                                          num_iter=3000,
                                                          lr=0.1, dis=True,
                                                          lam=0,
                                                          T_known=T_constant)


# # Use all readings to learn Home factors

# In[59]:

tensor_copy = all_tensor.copy()
H_all, A_all, T_all, Hs, As, Ts, HATs, costs = learn_HAT_adagrad_graph(case, tensor_copy,
                                                            L,
                                                          a,
                                                          b,
                                                          num_iter=3000,
                                                          lr=0.1, dis=True,
                                                          lam=0,
                                                          T_known=T_constant)


# In[60]:

from sklearn.cluster import KMeans
X = {}
y_pred = {}
x1 = {}
x2 = {}
# for home factors learnt from aggregate readings
X[0] = H_agg.copy()
X[0] = X[0]/np.max(X[0])
y_pred[0] = KMeans(n_clusters=5, random_state=0).fit_predict(X[0][:, :12])
x1[0], x2[0] = (-np.var(X[0], axis=0)).argsort()[:2]
# for home factors learnt from all readings
X[1] = H_all.copy()
X[1] = X[1]/np.max(X[1])
y_pred[1] = KMeans(n_clusters=5, random_state=0).fit_predict(X[1][:, :12])
x1[1], x2[1] = (-np.var(X[1], axis=0)).argsort()[:2]


# import Set
# for aggregate readings
start_index = len(source_tensor)
target_agg_cluster = set(y_pred[0][start_index:])
source_agg_index = [i for i, j in enumerate(y_pred[0][:start_index]) if j in target_agg_cluster]
source_agg_sub_tensor = source_tensor[source_agg_index]
# for all readings
target_all_cluster = set(y_pred[1][start_index:])
source_all_index = [i for i, j in enumerate(y_pred[1][:start_index]) if j in target_all_cluster]
source_all_sub_tensor = source_tensor[source_all_index]
# for intersection
source_inter_index = list(set(source_agg_index).intersection(source_all_index))
source_inter_sub_tensor = source_tensor[source_inter_index]
# for difference
source_diff_index = list(set(source_agg_index).difference(source_all_index))
source_diff_sub_tensor = source_tensor[source_diff_index]
# for union
source_union_index = list(set(source_agg_index).union(source_all_index))
source_union_sub_tensor = source_tensor[source_union_index]

import random
k = list((set(list(range(533))) - set(source_inter_index)))
random.seed(10)
random.shuffle(k)
source_random_index = source_inter_index + k[:11]
source_random_sub_tensor = source_tensor[source_random_index]


# In[122]:

print "agg subset cluster: ", target_agg_cluster
print "all subset cluster: ", target_all_cluster
print "agg subset size: ", len(source_agg_index)
print "all subset size: ", len(source_all_index)
print "intersection size: ", len(set(source_agg_index).intersection(source_all_index))


# In[123]:

# tensor_copy = source_tensor.copy()
# H_all, A_all, T_all, Hs_all, As_all, Ts_all, HATs_all, costs_all = learn_HAT_adagrad_graph(case, tensor_copy,
#                                                             source_L,
#                                                           a,
#                                                           b,
#                                                           num_iter=3000,
#                                                           lr=0.1, dis=True,
#                                                           lam=0,
#                                                           T_known=T_constant)


# In[79]:

# source_agg_sub_static = source_static[source_agg_index]
# source_agg_sub_L = get_L(source_agg_sub_static)
# source_all_sub_static = source_static[source_all_index]
# source_all_sub_L = get_L(source_all_sub_static)
# source_inter_sub_static = source_static[source_inter_index]
# source_inter_sub_L = get_L(source_inter_sub_static)
# source_diff_sub_static = source_static[source_diff_index]
# source_diff_sub_L = get_L(source_diff_sub_static)
source_random_sub_static = source_static[source_random_index]
source_random_sub_L = get_L(source_random_sub_static)

source_union_sub_static = source_static[source_union_index]
source_union_sub_L = get_L(source_union_sub_static)


# In[80]:
# tensor_copy = source_diff_sub_tensor.copy()
# H_sub_diff, A_sub_diff, T_sub_diff, Hs, As, Ts, HATs, costs = learn_HAT_adagrad_graph(case, tensor_copy, source_diff_sub_L, a, b, num_iter=3000, lr=0.1, dis=True,lam=0,T_known=T_constant)

# tensor_copy = source_random_sub_tensor.copy()
# H_sub_diff, A_random, T_sub_diff, Hs, As, Ts, HATs, costs = learn_HAT_adagrad_graph(case, tensor_copy, source_random_sub_L, a, b, num_iter=3000, lr=0.1, dis=True,lam=0,T_known=T_constant)


tensor_copy = source_union_sub_tensor.copy()
H_sub_diff, A_random, T_sub_diff, Hs, As, Ts, HATs, costs = learn_HAT_adagrad_graph(case, tensor_copy, source_union_sub_L, a, b, num_iter=3000, lr=0.1, dis=True,lam=0,T_known=T_constant)

print "length: ", len(H_sub_diff)
# tensor_copy = source_all_sub_tensor.copy()
# H_sub_all, A_sub_all, T_sub_all, Hs_sub_all, As_sub_all, Ts_sub_all, HATs_sub_all, costs_sub_all = learn_HAT_adagrad_graph(
#                                                             case, tensor_copy,
#                                                             source_all_sub_L,
#                                                           a,
#                                                           b,
#                                                           num_iter=3000,
#                                                           lr=0.1, dis=True,
#                                                           lam=0,
#                                                           T_known=T_constant)



# # In[81]:


# tensor_copy = source_agg_sub_tensor.copy()
# H_sub_agg, A_sub_agg, T_sub_agg, Hs_sub_agg, As_sub_agg, Ts_sub_agg, HATs_sub_agg, costs_sub_agg = learn_HAT_adagrad_graph(
#                                                         case, tensor_copy,
#                                                             source_agg_sub_L,
#                                                           a,
#                                                           b,
#                                                           num_iter=3000,
#                                                           lr=0.1, dis=True,
#                                                           lam=0,
#                                                           T_known=T_constant)

# tensor_copy = source_inter_sub_tensor.copy()
# H_sub_inter, A_sub_inter, T_sub_inter, Hs_sub_inter, As_sub_inter, Ts_sub_inter, HATs_sub_inter, costs_sub_inter = learn_HAT_adagrad_graph(
#                                                         case, tensor_copy,
#                                                             source_inter_sub_L,
#                                                           a,
#                                                           b,
#                                                           num_iter=3000,
#                                                           lr=0.1, dis=True,
#                                                           lam=0,
#                                                           T_known=T_constant)


# In[82]:

from scipy.optimize import nnls

n_splits = 10
case = 2
a = 3
b = 3
cost = 'abs'
start = 1
stop = 13


pred_all = {}
pred_sub_all = {}
pred_sub_agg = {}
pred_sub_inter = {}
pred_sub_diff = {}

for random_seed in range(5):
    
    pred_all[random_seed] = {}
    pred_sub_all[random_seed] = {}
    pred_sub_agg[random_seed] = {}
    pred_sub_inter[random_seed] = {}
    pred_sub_diff[random_seed] = {}
    
    for appliance in APPLIANCES_ORDER:
        pred_all[random_seed][appliance] = {f:[] for f in range(10, 110, 10)}
        pred_sub_all[random_seed][appliance] = {f:[] for f in range(10, 110, 10)}
        pred_sub_agg[random_seed][appliance] = {f:[] for f in range(10, 110, 10)}
        pred_sub_inter[random_seed][appliance] = {f:[] for f in range(10, 110, 10)}
        pred_sub_diff[random_seed][appliance] = {f:[] for f in range(10, 110, 10)}



kf = KFold(n_splits=n_splits)

for random_seed in range(5):
    print "random seed: ", random_seed
    for train_percentage in range(10, 100, 20):
        print "training percentage: ", train_percentage
        rd = 0
        for train_max, test in kf.split(target_df):
                print "round: ", rd


                num_train = int((train_percentage*len(train_max)/100)+0.5)
                num_test = len(test)

                # get the random training data from train_max based on then random seed
                if train_percentage==100:
                    train = train_max
                else:
                    train, _ = train_test_split(train_max, train_size = train_percentage/100.0, random_state=random_seed)

                # get the index of training and testing data
                train_ix = target_df.index[train]
                test_ix = target_df.index[test]
                print "test_ix: ", test_ix
                print "training set length: ", len(train_ix)

                # create the tensor
                train_test_ix = np.concatenate([test_ix, train_ix])
                df_t, dfc_t = target_df.ix[train_test_ix], target_dfc.ix[train_test_ix]
                tensor = get_tensor(df_t, start, stop)
#                 print tensor.shape
                L_inner = target_L[np.ix_(np.concatenate([test, train]), np.concatenate([test, train]))]


                ############################################################################################
                # Transfer learning: with A_all learned with all Austin Data
                ############################################################################################
                # tensor_copy = tensor.copy()
                # tensor_copy[:num_test, 1:, :] = np.NaN
                # H, A, T, Hs, As, Ts, HATs, costs = learn_HAT_adagrad_graph(case, tensor_copy,
                #                                             L_inner,
                #                                           a,
                #                                           b,
                #                                           num_iter=2000,
                #                                           lr=0.1, dis=False,
                #                                           lam=0,
                #                                           A_known = A_all,
                #                                           T_known=T_constant)
                # # get the prediction
                # HAT = multiply_case(H, A, T, case)
                # for appliance in APPLIANCES_ORDER:
                #     pred_all[random_seed][appliance][train_percentage].append(pd.DataFrame(HAT[:num_test, appliance_index[appliance], :], index=test_ix))


                #############################################################################################
                # transfer learning: with A_sub_agg learned with subset of Austin Data
                ############################################################################################
                # tensor_copy = tensor.copy()
                # tensor_copy[:num_test, 1:, :] = np.NaN
                # H, A, T, Hs, As, Ts, HATs, costs = learn_HAT_adagrad_graph(case, tensor_copy,
                #                                             L_inner,
                #                                           a,
                #                                           b,
                #                                           num_iter=2000,
                #                                           lr=0.1, dis=False,
                #                                           lam=0,
                #                                           A_known = A_sub_agg,
                #                                           T_known=T_constant)
                # # get the prediction
                # HAT = multiply_case(H, A, T, case)
                # for appliance in APPLIANCES_ORDER:
                #     pred_sub_agg[random_seed][appliance][train_percentage].append(pd.DataFrame(HAT[:num_test, appliance_index[appliance], :], index=test_ix))

                #############################################################################################
                # transfer learning: with A_sub_all learned with subset of Austin Data
                ############################################################################################
                # tensor_copy = tensor.copy()
                # tensor_copy[:num_test, 1:, :] = np.NaN
                # H, A, T, Hs, As, Ts, HATs, costs = learn_HAT_adagrad_graph(case, tensor_copy,
                #                                             L_inner,
                #                                           a,
                #                                           b,
                #                                           num_iter=2000,
                #                                           lr=0.1, dis=False,
                #                                           lam=0,
                #                                           A_known = A_sub_all,
                #                                           T_known=T_constant)
                # # get the prediction
                # HAT = multiply_case(H, A, T, case)
                # for appliance in APPLIANCES_ORDER:
                #     pred_sub_all[random_seed][appliance][train_percentage].append(pd.DataFrame(HAT[:num_test, appliance_index[appliance], :], index=test_ix))


                #############################################################################################
                # transfer learning: with A_sub_inter learned with subset of Austin Data
                ############################################################################################
                # tensor_copy = tensor.copy()
                # tensor_copy[:num_test, 1:, :] = np.NaN
                # H, A, T, Hs, As, Ts, HATs, costs = learn_HAT_adagrad_graph(case, tensor_copy,
                #                                             L_inner,
                #                                           a,
                #                                           b,
                #                                           num_iter=2000,
                #                                           lr=0.1, dis=False,
                #                                           lam=0,
                #                                           A_known = A_sub_inter,
                #                                           T_known=T_constant)
                # # get the prediction
                # HAT = multiply_case(H, A, T, case)
                # for appliance in APPLIANCES_ORDER:
                #     pred_sub_inter[random_seed][appliance][train_percentage].append(pd.DataFrame(HAT[:num_test, appliance_index[appliance], :], index=test_ix))


                #############################################################################################
                # transfer learning: with A_sub_diff learned with subset of Austin Data
                ############################################################################################
                tensor_copy = tensor.copy()
                tensor_copy[:num_test, 1:, :] = np.NaN
                H, A, T, Hs, As, Ts, HATs, costs = learn_HAT_adagrad_graph(case, tensor_copy,
                                                            L_inner,
                                                          a,
                                                          b,
                                                          num_iter=2000,
                                                          lr=0.1, dis=False,
                                                          lam=0,
                                                          A_known = A_random,
                                                          T_known=T_constant)
                # get the prediction
                HAT = multiply_case(H, A, T, case)
                for appliance in APPLIANCES_ORDER:
                    pred_sub_diff[random_seed][appliance][train_percentage].append(pd.DataFrame(HAT[:num_test, appliance_index[appliance], :], index=test_ix))



                rd += 1


# In[83]:
out = {}
for random_seed in range(5):
  out[random_seed] = {}
  for appliance in APPLIANCES_ORDER[1:]:
    out[random_seed][appliance] = {}
    for f in range(10, 110, 20):
      s = pd.concat(pred_sub_diff[random_seed][appliance][f]).loc[target_df.index]
      if appliance  == 'hvac':
        out[random_seed][appliance][f] = compute_rmse_fraction(appliance, s[range(4, 10)], 'SanDiego')[2]
      else:
        out[random_seed][appliance][f] = compute_rmse_fraction(appliance, s, 'SanDiego')[2]


# In[101]:
mean_out = pd.Panel(out).mean(axis=0)


import pickle
pickle.dump(pred_sub_diff, open('./pred_sub_union.pkl', 'w'))
pickle.dump(out, open('./out_sub_union.pkl', 'w'))
