
# coding: utf-8

# In[1]:


import sys

# from degree_days import dds
from tensor_custom_core import *
sys.path.insert(0, '../../aaai18/code/')
import datetime
from sklearn.model_selection import train_test_split, KFold
from common import *
from create_matrix import *
from sklearn.metrics.pairwise import cosine_similarity

import random
from sklearn.cluster import KMeans

appliance_index = {appliance: APPLIANCES_ORDER.index(appliance) for appliance in APPLIANCES_ORDER}

APPLIANCES = ['fridge', 'hvac', 'wm', 'mw', 'oven', 'dw']
region = "Austin"
year = 2014

import os

def un_normalize(x, maximum, minimum):
    return (maximum - minimum) * x + minimum


# In[2]:


df, dfc, tensor, static_feature = create_region_df_dfc_static('Austin', 2014, 1, 13)
L = get_L(static_feature)
case = 2
a = 3
b = 3
weight_matrix = np.ones(tensor.shape)
T_constant = np.ones(12).reshape(-1, 1)
tensor_copy = tensor.copy()
H, A, T, Hs, As, Ts, HATs, costs = learn_HAT_adagrad_graph(case, tensor_copy,
                                                            L,
                                                          a,
                                                          b, weight_matrix,
                                                          num_iter=600,
                                                          lr=0.5, dis=False,
                                                          lam=0,
                                                          T_known=T_constant)


# In[3]:


from sklearn.cluster import KMeans
# for home factors learnt from aggregate readings
X = H.copy()
X = X/np.max(X)
cluster = KMeans(n_clusters=10, random_state=0).fit_predict(X)
x1, x2 = (-np.var(X, axis=0)).argsort()[:2]

x1=2 
x2=0


# In[4]:


import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

start = len(tensor)
fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.scatter(X[:, x1], X[:, x2], c = cluster, cmap='Set2', lw=0)
# plt.xlabel("H1")
# plt.ylabel("H2")
# plt.title("home factors learnt from all readings")


target_cluster_id = 4
target_cluster_idx = [i for i, j in enumerate(cluster) if j == target_cluster_id]


# In[5]:


import random
random.seed(10)
random.shuffle(target_cluster_idx)
test_idx = target_cluster_idx[:6]
validate_idx = target_cluster_idx[6:12]
test_validate_idx = np.r_[test_idx, validate_idx]
train_idx = list(set(list(range(533))) - set(test_validate_idx))


# In[6]:


train_ix = df.index[train_idx]
test_ix = df.index[test_idx]
validate_ix = df.index[validate_idx]
idx = np.r_[test_ix, validate_ix, train_ix]
df_t = df.loc[idx]
tensor = get_tensor(df_t, 1, 13)
L_inner = L[np.ix_(np.r_[test_validate_idx, train_idx], np.r_[test_validate_idx, train_idx])]


# In[7]:


from sklearn.metrics.pairwise import cosine_similarity


# In[8]:


num_home_factor = 3
num_season_factor = 3

num_test = len(test_ix)
num_validate = len(validate_ix)
num_train = len(train_ix)

num_iterations = 10


train_df = df.loc[train_ix]
test_validate_df = df.loc[np.r_[test_ix, validate_ix]]

train_tensor = get_tensor(train_df, 1, 13)
test_validate_tensor = get_tensor(test_validate_df, 1, 13)

L_train = L[np.ix_(train_idx, train_idx)]
L_test_validate = L[np.ix_(test_validate_idx, test_validate_idx)]


weight_matrices = {}
train_weight = np.ones(train_tensor.shape)
origin_weight = np.ones(train_tensor.shape)

H = {}
A = {}
T = {}

choose = "max"

pred = {}
pred_validation = {}
error = {}
for iterations in range(num_iterations):
    pred[iterations] = {}
    pred_validation[iterations] = {}
    error[iterations] = {}
    for appliance in APPLIANCES_ORDER:
        pred[iterations][appliance] = []
        pred_validation[iterations][appliance] = []
        error[iterations][appliance] = []

tv_weight = np.ones(test_validate_tensor.shape)
print tv_weight.shape

for iteration in range(num_iterations):
    print "iteration: ", iteration
    
    # print train_weight
    weight_matrices[iteration] = train_weight
    tensor_copy = train_tensor.copy()

    # do tensor factorization
    H_train, A_train, T_train, Hs, As, Ts, HATs, costs = learn_HAT_adagrad_graph(case, tensor_copy,
                                                               L_train,
                                                              num_home_factor,
                                                              num_season_factor,
                                                             train_weight,
                                                              num_iter=600,
                                                              lr=0.5, dis=False,
                                                              lam=0,
                                                              T_known=T_constant)

    # use A, T to learn Home factors of validate and test homes
    HAT_train = multiply_case(H_train, A_train, T_train, 2)
    tensor_copy = test_validate_tensor.copy()
    tensor_copy[:num_test, 1:, :] = np.NaN
    
    
    H_tv, A_tv, T_tv, Hs, As, Ts, HATs, costs = learn_HAT_adagrad_graph(case, tensor_copy, L_test_validate,
                                                                       num_home_factor, num_season_factor, tv_weight,
                                                                       num_iter=600, lr=0.5, dis=False, lam=0,
                                                                       A_known=A_train, T_known=T_train)
    
    HAT_tv = multiply_case(H_tv, A_tv, T_tv, 2)
    for appliance in APPLIANCES_ORDER:
        pred[iteration][appliance].append(pd.DataFrame(HAT_tv[:num_test, appliance_index[appliance], :], index = test_ix))
        pred_validation[iteration][appliance].append(pd.DataFrame(HAT_tv[num_test:, appliance_index[appliance], :], index = validate_ix))
    
    # compute the appliance prediction error for each validation homes
    for appliance in APPLIANCES_ORDER:
        s = pd.concat(pred_validation[iteration][appliance]).loc[validate_ix]
        if appliance=="hvac":
            error[iteration][appliance] = compute_rmse_fraction(appliance,s[range(4, 10)], 'Austin')[3]
        else:   
            error[iteration][appliance] = compute_rmse_fraction(appliance, s, 'Austin')[3]
    
#     print error[iteration]['hvac']
    error[iteration]['aggregate'] = error[iteration]['aggregate'][error[iteration]['aggregate'].index.duplicated()]

    # compute the average appliance error for each home
    error_avg = {}
    for appliance in APPLIANCES_ORDER:

        if appliance == 'hvac':
            start, end = 5, 11
        else:
            start, end = 1, 13

        error_home = pd.concat([error[iteration][appliance][appliance + "_{}".format(start)], 
                           error[iteration][appliance][appliance + "_{}".format(start+1)]],axis=1)

        for i in range(start+2, end):
            error_home = pd.concat([error_home, error[iteration][appliance][appliance + "_{}".format(i)]], axis=1)
        app = np.sqrt((error_home**2).mean(axis=1))
        error_avg[appliance] = app
        
    
    # leave one out to calculate the weight for each training home's appliance
    difference = {}
    error_inner = {}
    pred_inner = {}
    pred_validation_inner = {}
    for i in range(num_train):
        pred_inner[i] = {}
        pred_validation_inner[i] = {}
        print "traning home: ", i
        tensor_copy = train_tensor.copy()
        tensor_copy[i, :, :] = np.NaN
        
        H_train_inner, A_train_inner, T_train_inner, Hs, As, Ts, HATs, costs = learn_HAT_adagrad_graph(case, tensor_copy,
                                                                                                   L_train,
                                                                                                  num_home_factor,
                                                                                                  num_season_factor,
                                                                                                 origin_weight,
                                                                                                  num_iter=600,
                                                                                                  lr=0.5, dis=False,
                                                                                                  lam=0,
                                                                                                  T_known=T_constant)
        

        HAT_train_inner = multiply_case(H_train_inner, A_train_inner, T_train_inner, 2)
        tensor_copy = test_validate_tensor.copy()
        tensor_copy[:num_test, 1:, :] = np.NaN


        H_tv_inner, A_tv_inner, T_tv_inner, Hs, As, Ts, HATs, costs = learn_HAT_adagrad_graph(case, tensor_copy, L_test_validate,
                                                                           num_home_factor, num_season_factor, tv_weight,
                                                                           num_iter=600, lr=0.5, dis=False, lam=0,
                                                                           A_known=A_train_inner, T_known=T_train_inner)

        HAT_tv_inner = multiply_case(H_tv_inner, A_tv_inner, T_tv_inner, 2)
        for appliance in APPLIANCES_ORDER:
            pred_inner[i][appliance] = []
            pred_validation_inner[i][appliance] = []
            pred_inner[i][appliance].append(pd.DataFrame(HAT_tv_inner[:num_test, appliance_index[appliance], :], index = test_ix))
            pred_validation_inner[i][appliance].append(pd.DataFrame(HAT_tv_inner[num_test:, appliance_index[appliance], :], index = validate_ix))
    
        
        error_inner[i] = {}
        # compute the appliance prediction error for each validation homes
        for appliance in APPLIANCES_ORDER:
            s = pd.concat(pred_validation_inner[i][appliance]).loc[validate_ix]
            if appliance=="hvac":
                error_inner[i][appliance] = compute_rmse_fraction(appliance,s[range(4, 10)], 'Austin')[3]
            else:   
                error_inner[i][appliance] = compute_rmse_fraction(appliance, s, 'Austin')[3]
        
        error_inner[i]['aggregate'] = error_inner[i]['aggregate'][error_inner[i]['aggregate'].index.duplicated()]
        
        # compute the average appliance error for each home
        error_avg_inner = {}
        for appliance in APPLIANCES_ORDER:

            if appliance == 'hvac':
                start, end = 5, 11
            else:
                start, end = 1, 13

            error_home_inner = pd.concat([error_inner[i][appliance][appliance + "_{}".format(start)], 
                               error_inner[i][appliance][appliance + "_{}".format(start+1)]],axis=1)

            for idx in range(start+2, end):
                error_home = pd.concat([error_home_inner, error_inner[i][appliance][appliance + "_{}".format(idx)]], axis = 1)
            app = np.sqrt((error_home_inner**2).mean(axis=1))
            error_avg_inner[appliance] = app
        
        # calculate the difference between origin and the leave-one-out setting
        difference[i] = ((pd.DataFrame(error_avg_inner) - pd.DataFrame(error_avg))/pd.DataFrame(error_avg)).max()
    
    train_weight = np.ones([num_train, 7, 12])
    for i in range(num_train):
        for appliance in APPLIANCES_ORDER:
            train_weight[i, appliance_index[appliance], :] = difference[i][appliance]

    H[iteration] = np.r_[H_tv, H_train]
    A[iteration] = A_train
    T[iteration] = T_train

import pickle
pickle.dump(pred, open("./results/pred_loo.pkl", 'w'))
pickle.dump(pred_validation, open("./results/pred_validation_loo.pkl", 'w'))
pickle.dump(H, open("./results/H_loo.pkl", 'w'))
pickle.dump(A, open("./results/A_loo.pkl", 'w'))
pickle.dump(T, open("./results/T_loo.pkl", 'w'))
pickle.dump(weight_matrices, open("./results/weight_matrices_loo.pkl", 'w'))


