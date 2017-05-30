from mf_core import *

import numpy as np
import pandas as pd
import sys
import os
from create_matrix import *
from sklearn.model_selection import LeaveOneOut
from sklearn.neighbors import KNeighborsRegressor
import pickle
from scipy.io import savemat

region = "Austin"
year = 2014

import sys
all_features = True
out = {}
for appliance in ['fridge','hvac','dw','mw','oven','wm']:
	appliance_df = create_matrix_region_appliance_year(region, year, appliance, all_features=all_features)
	aggregate_cols = [x for x in appliance_df.columns if "aggregate" in x]
	appliance_cols = [x for x in appliance_df.columns if appliance in x]
	energy_cols = np.concatenate([aggregate_cols, appliance_cols])

	df = appliance_df.copy()
	dfc = df.copy()

	df = df[energy_cols]
	out[appliance] = df.values

savemat('data/all_features.mat', out)