
# coding: utf-8

# In[1]:

from sklearn.model_selection import train_test_split, KFold
import sys
from tensor_custom_core import *
sys.path.insert(0, '../../aaai18/code/')
from common import *
from create_matrix import *
import random
from sklearn.metrics.pairwise import cosine_similarity
import multiprocessing as mp

global case, num_home_factor, num_season_factor
global start, stop, T_constant
global source_df, source_dfc, source_tensor, source_static
global num_iterations

# initialization
appliance_index = {appliance: APPLIANCES_ORDER.index(appliance) for appliance in APPLIANCES_ORDER}
APPLIANCES = ['fridge', 'hvac', 'wm', 'mw', 'oven', 'dw']
region = "Austin"
year = 2014
case = 2
num_home_factor = 3
num_season_factor = 3
source = 'Austin'
target = 'SanDiego'
constant_use = 'True'
start = 1
stop = 13
T_constant = np.ones(12).reshape(-1, 1)

source_df, source_dfc, source_tensor, source_static = create_region_df_dfc_static(source, year, start, stop)
target_df, target_dfc, target_tensor, target_static = create_region_df_dfc_static(target, year, start, stop)

test_size = 50
num_iterations = 10

def compute_each_home(index):
    print index
    print "num_iterations: ", num_iterations
    weight_matrices = {}
    pred_train = {}
    pred = {}
    error = {}
    H_history = {}

    for iters in range(num_iterations):
        pred_train[iters] = {}
        pred[iters] = {}
        error[iters] = {}
        for appliance in APPLIANCES_ORDER:
            pred_train[iters][appliance] = []
            pred[iters][appliance] = []
            error[iters][appliance] = []

    test_df = source_df.iloc[[index]]
    train_df = source_df.drop(source_df.index[[index]])
    
    train_index = train_df.index
    test_index = test_df.index

    df = pd.concat([test_df, train_df])
    tensor = get_tensor(df, 1, 13)

    train_weight = np.ones(tensor.shape)
        
    for iters in range(num_iterations):
        
        print 'iteration: ', iters
        weight_matrices[iters] = train_weight 
        tensor_copy = tensor.copy()
        tensor_copy[0, 1:, :] = np.NaN
        
        H, A, T, Hs, As, Ts, HATs, costs = learn_HAT_adagrad_graph(case, tensor_copy,
                                                               None,
                                                              num_home_factor,
                                                              num_season_factor,
                                                             train_weight,
                                                              num_iter=2000,
                                                              lr=0.1, dis=True,
                                                              lam=0,
                                                              T_known=T_constant)
        
        # get the similarity between home factors
        similarities = cosine_similarity(H)
        sim = similarities[0]

        # get the prediction error of training homes
        HAT = multiply_case(H, A, T, case)
        for appliance in APPLIANCES_ORDER:
            pred_train[iters][appliance].append(pd.DataFrame(HAT[1:, appliance_index[appliance], :], index=train_index))
            pred[iters][appliance].append(pd.DataFrame([HAT[0, appliance_index[appliance], :]], index=test_index))
       
        # compute the appliance prediction error for each training homes
        for appliance in APPLIANCES_ORDER[1:]:
            s = pd.concat(pred_train[iters][appliance]).loc[train_index]
            if appliance=="hvac":
                error[iters][appliance] = compute_rmse_fraction(appliance,s[range(4, 10)], 'Austin')[3]
            else:   
                error[iters][appliance] = compute_rmse_fraction(appliance, s, 'Austin')[3]
        
        # compute the average appliance error for each home
        error_avg = {}
        for appliance in APPLIANCES_ORDER[1:]:

            if appliance == 'hvac':
                start, end = 5, 11
            else:
                start, end = 1, 13

            error_home = pd.concat([error[iters][appliance][appliance + "_{}".format(start)], 
                               error[iters][appliance][appliance + "_{}".format(start+1)]],axis=1)

            for k in range(start+2, end):
                error_home = pd.concat([error_home, error[iters][appliance][appliance + "_{}".format(k)]], axis=1)
            app = np.sqrt((error_home**2).mean(axis=1))
            error_avg[appliance] = app
        error_overall = (pd.DataFrame(error_avg).fillna(0)*pd.Series(contri['Austin'])).sum(axis=1)
        train_confidence = 1/pd.DataFrame(error_overall).reindex(train_index).T.as_matrix()

        sim = (sim - sim.min())/(sim.max() - sim.min())
        train_weight = np.repeat(sim.reshape(-1, 1), 12*7, axis=1).reshape(-1, 7, 12)
        H_history[iters] = H
    
    return pred

###################################################
results = []
cpus = mp.cpu_count()
pool = mp.Pool(53)
for i in range(533):
    # print i
    result = pool.apply_async(compute_each_home, args=(i,))
    results.append(result)
pool.close()
pool.join()
###################################################

# get the results of all processes
prediction = {}
i = 0
for result in results:
    prediction[i] = result.get()
    i += 1

import pickle
pickle.dump(prediction, open('./results/prediciton.pkl', 'w'))
# print rexs

# import pickle
# pickle.dump(pred_test, open('./results/pred_all_loo.pkl', 'w'))
# pickle.dump(pred, open('./results/pred_single_loo.pkl', 'w'))
