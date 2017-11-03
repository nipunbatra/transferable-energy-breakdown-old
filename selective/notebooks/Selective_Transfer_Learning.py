
# coding: utf-8

# In[10]:

import sys


# In[11]:

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


# In[12]:

df, dfc, tensor, static = create_region_df_dfc_static(region, year)
au_df, au_dfc, au_tensor, au_static = create_region_df_dfc_static("Austin", 2014)
sd_df, sd_dfc, sd_tensor, sd_static = create_region_df_dfc_static("SanDiego", 2014)

au_L = get_L(au_static)
sd_L = get_L(sd_static)
T_constant = np.ones(12).reshape(-1, 1)


# In[ ]:

n_splits = 10
kf = KFold(n_splits=n_splits)

num_home_factor = 3
num_season_factor = 3
case = 2
prediction = {}
num_iterations = 10
num_random_seed = 5

for random_seed in range(num_random_seed):
    prediction[random_seed] = {}
    for iterations in range(num_iterations):
        prediction[random_seed][iterations] = {}
        for appliance in APPLIANCES_ORDER:
            prediction[random_seed][iterations][appliance] = []
        
# Weight matrix for Austin Data
m, n, o = au_tensor.shape
weight_matrix = np.ones((m,n,o))

adapt_percentage = 50
for random_seed in range(num_random_seed):
    random.seed(random_seed)
    
    for adapt_max, test in kf.split(sd_df):
        
        np.random.seed(10*random_seed + int(adapt_percentage)/10)
        np.random.shuffle(adapt_max)
        
        
        # prepare for the adapt data and test data in SanDiego
        num_test = len(test)
        num_adapt = int((adapt_percentage * len(adapt_max) / 100) + 0.5)

        if adapt_percentage == 100:
            adapt = adapt_max 
        else:
            adapt, _ = train_test_split(adapt_max, train_size=adapt_percentage / 100.0)

        adapt_ix = sd_df.index[adapt]
        test_ix = sd_df.index[test]

        weight_matrix = np.ones((m,n,o))
       

        for iteration in range(num_iterations):
            print 'iteration:', iteration
#             print weight_matrix
            # train A_au with au_tensor
            tensor_copy = au_tensor.copy()
            H_train, A_train, T_train, Hs, As, Ts, HATs, costs = learn_HAT_adagrad_graph(case, tensor_copy,
                                                                       au_L,
                                                                      num_home_factor,
                                                                      num_season_factor,
                                                                     weight_matrix,
                                                                      num_iter=2000,
                                                                      lr=0.1, dis=True,
                                                                      lam=0,
                                                                      T_known=T_constant)

            # create the adapt-test tensor for SanDiego
            test_adapt_ix = np.concatenate([test_ix, adapt_ix])
            ta_df = sd_df.loc[test_adapt_ix]
            ta_tensor = get_tensor(ta_df)

            weight_ta = np.ones(ta_tensor.shape)
            L_ta = sd_L[np.ix_(np.concatenate([test, adapt]), np.concatenate([test, adapt]))]

            # adapt the A_train to SanDiego
            tensor_copy = ta_tensor.copy()
            H_ta, A_ta, T_ta, Hs, As, Ts, HATs, costs = learn_HAT_adagrad_graph(case, tensor_copy,
                                                                       L_ta,
                                                                      num_home_factor,
                                                                      num_season_factor,
                                                                     weight_ta,
                                                                      num_iter=2000,
                                                                      lr=0.1, dis=True,
                                                                      lam=0,
                                                                    A_known = A_train,
                                                                      T_known=T_constant)



            # save the result for this iteration.
            HAT = multiply_case(H_ta, A_ta, T_ta, case)
            for appliance in APPLIANCES_ORDER:
                prediction[random_seed][iteration][appliance].append(pd.DataFrame(HAT[:num_test, appliance_index[appliance], :], index = test_ix))

            # update weight matrix for training data
            H_tr_v = np.r_[H_train, H_ta[num_test:]]

            similarities = cosine_similarity(H_tr_v)
            sim = similarities[:len(H_train), len(H_train):]
            sim = sim.max(axis=1)
            
            weight = (sim-sim.min())/(sim.max() - sim.min())
#             print weight
            weight_matrix = np.repeat(weight.reshape(-1, 1), 12*7, axis=1).reshape(-1, 7, 12)


# In[6]:

out = {}
for random_seed in range(num_random_seed):
    out[random_seed] = {}
    for iteration in range(num_iterations):
        out[random_seed][iteration] = {}
        for appliance in APPLIANCES_ORDER[1:]:
            out[random_seed][iteration][appliance] = {}

            print random_seed, adapt_percentage, iteration, appliance
            s = pd.concat(prediction[random_seed][iteration][appliance]).loc[sd_df.index]
            if appliance=="hvac":
                out[random_seed][iteration][appliance] = compute_rmse_fraction(appliance,s[range(4, 10)],'SanDiego')[2]
            else:   
                out[random_seed][iteration][appliance] = compute_rmse_fraction(appliance, s, 'SanDiego')[2]


import pickle
pickle.dump(prediction, open('./prediction_{}_{}.pkl'.format(num_home_factor, num_season_factor), 'w'))
pickle.dump(out, open('./out_{}_{}.pkl'.format(num_home_factor, num_season_factor), 'w'))