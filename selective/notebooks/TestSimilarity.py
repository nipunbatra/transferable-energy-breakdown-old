
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
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
                                                          num_iter=3000,
                                                          lr=0.1, dis=True,
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
ax = fig.add_subplot(111)
ax.scatter(X[:, x1], X[:, x2], c = cluster, cmap='Set2', lw=0)
plt.xlabel("H1")
plt.ylabel("H2")
plt.title("home factors learnt from all readings")

target_cluster_id = 4
target_cluster_idx = [i for i, j in enumerate(cluster) if j == target_cluster_id]
# ax.scatter(X[target_cluster_idx, x1], X[target_cluster_idx, x2], color='black', marker='o', facecolors='none')
print len(target_cluster_idx)


# In[5]:


len(target_cluster_idx)


# In[6]:


import random
random.seed(10)
random.shuffle(target_cluster_idx)
test_idx = target_cluster_idx[:6]
validate_idx = target_cluster_idx[6:12]
test_validate_idx = np.r_[test_idx, validate_idx]
train_idx = list(set(list(range(533))) - set(test_validate_idx))


# In[7]:


train_ix = df.index[train_idx]
test_ix = df.index[test_idx]
validate_ix = df.index[validate_idx]
idx = np.r_[test_ix, validate_ix, train_ix]
df_t = df.loc[idx]
tensor = get_tensor(df_t, 1, 13)
L_inner = L[np.ix_(np.r_[test_validate_idx, train_idx], np.r_[test_validate_idx, train_idx])]


# In[8]:


from sklearn.metrics.pairwise import cosine_similarity


# In[ ]:


num_home_factor = 3
num_season_factor = 3

num_test = len(test_ix)
num_validate = len(validate_ix)
num_train = len(train_ix)

num_iterations = 20

train_df = df.loc[train_ix]
test_validate_df = df.loc[np.r_[test_ix, validate_ix]]

train_tensor = get_tensor(train_df, 1, 13)
test_validate_tensor = get_tensor(test_validate_df, 1, 13)

L_train = L[np.ix_(train_idx, train_idx)]
L_test_validate = L[np.ix_(test_validate_idx, test_validate_idx)]


weight_matrices = {}
train_weight = np.ones(train_tensor.shape)

H = {}
A = {}
T = {}

choose = "max"

pred = {}
pred_validation = {}
for iterations in range(num_iterations):
    pred[iterations] = {}
    pred_validation[iterations] = {}
    for appliance in APPLIANCES_ORDER:
        pred[iterations][appliance] = []
        pred_validation[iterations][appliance] = []

tv_weight = np.ones(test_validate_tensor.shape)
print tv_weight.shape

for iteration in range(num_iterations):
    print "iteration: ", iteration
    
    weight_matrices[iteration] = train_weight
    tensor_copy = train_tensor.copy()

    # do tensor factorization
    H_train, A_train, T_train, Hs, As, Ts, HATs, costs = learn_HAT_adagrad_graph(case, tensor_copy,
                                                               L_train,
                                                              num_home_factor,
                                                              num_season_factor,
                                                             train_weight,
                                                              num_iter=3000,
                                                              lr=0.1, dis=True,
                                                              lam=0,
                                                              T_known=T_constant)

    # use A, T to learn Home factors of validate and test homes
    tensor_copy = test_validate_tensor.copy()
    tensor_copy[:num_test, 1:, :] = np.NaN
    print tensor_copy.shape
    
    
    H_tv, A_tv, T_tv, Hs, As, Ts, HATs, costs = learn_HAT_adagrad_graph(case, tensor_copy, L_test_validate,
                                                                       num_home_factor, num_season_factor, tv_weight,
                                                                       num_iter=3000, lr=0.1, dis=True, lam=0,
                                                                       A_known=A_train, T_known=T_train)
    
    HAT = multiply_case(H_tv, A_tv, T_tv, 2)
    for appliance in APPLIANCES_ORDER:
        pred[iteration][appliance].append(pd.DataFrame(HAT[:num_test, appliance_index[appliance], :], index = test_ix))
        pred_validation[iteration][appliance].append(pd.DataFrame(HAT[num_test:, appliance_index[appliance], :], index = validate_ix))
                
    # use the Home factors to update the weight matrix
    H_validate = H_tv[num_test:]
    
    H_validate_train = np.r_[H_validate, H_train]
    similarities = cosine_similarity(H_validate_train)
    sim = similarities[num_validate:, :num_validate]
    ####
    sim = sim.min(axis=1)
    sim = (sim - sim.min())/(sim.max() - sim.min())
    sim[200:] = 0
    ####
    # print len
    train_weight = np.repeat(sim.reshape(-1, 1), 12*7, axis=1).reshape(-1, 7, 12)
    print train_weight.shape
    
    # store the home factors
    H[iteration] = np.r_[H_tv, H_train]
    A[iteration] = A_train
    T[iteration] = T_train
    

import pickle
# pickle.dump(pred, open("./results/pred_max.pkl", 'w'))
# pickle.dump(pred_validation, open("./results/pred_validation_max.pkl", 'w'))
# pickle.dump(H, open("./results/H_max.pkl", 'w'))
# pickle.dump(A, open("./results/A_max.pkl", 'w'))
# pickle.dump(T, open("./results/T_max.pkl", 'w'))
# pickle.dump(weight_matrices, open("./results/weight_matrices_max.pkl", 'w'))

pickle.dump(pred, open("./results/pred_min.pkl", 'w'))
pickle.dump(pred_validation, open("./results/pred_validation_min.pkl", 'w'))
pickle.dump(H, open("./results/H_min.pkl", 'w'))
pickle.dump(A, open("./results/A_min.pkl", 'w'))
pickle.dump(T, open("./results/T_min.pkl", 'w'))
pickle.dump(weight_matrices, open("./results/weight_matrices_min.pkl", 'w'))
