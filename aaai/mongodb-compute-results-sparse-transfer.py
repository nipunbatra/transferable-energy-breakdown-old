from create_matrix import create_matrix_region_appliance_year
from subprocess import Popen
from common import compute_rmse_fraction
from create_matrix import *

from tensor_custom_core import *
from create_matrix import *
from tensor_custom_core import *
from degree_days import dds
from pymongo import MongoClient
appliance_index = {appliance: APPLIANCES_ORDER.index(appliance) for appliance in APPLIANCES_ORDER}

import os
import pickle
client = MongoClient()


source, target, static_fac, lam, num_home_factors, num_season_factors, random_seed, train_percentage, cost = sys.argv[1:]
name = "{}-{}-{}-{}-{}-{}-{}-{}-{}".format(source, target, static_fac, lam, num_home_factors, num_season_factors, random_seed, train_percentage, cost)
directory = os.path.expanduser('~/aaai2017/transfer_{}_{}_{}/'.format(source, target, cost))

filename = os.path.join(directory, name + '.pkl')
pr = pickle.load(open(filename, 'r'))
out = {'static':static_fac, 'lam':lam, 'num_home_factors':num_home_factors,
       'num_season_factors':num_season_factors,'train_percentage':int(train_percentage),
       'random_seed':random_seed}
for appliance in APPLIANCES_ORDER:
	if appliance=="hvac":
		out[appliance] = compute_rmse_fraction(appliance, pr[appliance][range(4, 10)], 'SanDiego')[2]
	else:
		out[appliance] = compute_rmse_fraction(appliance, pr[appliance], 'SanDiego')[2]

db = client.test_nipun
posts = db.posts
posts.insert_one(out)


