from pymongo import MongoClient

from common import compute_rmse_fraction
from create_matrix import *

appliance_index = {appliance: APPLIANCES_ORDER.index(appliance) for appliance in APPLIANCES_ORDER}

import os
import pickle

client = MongoClient()

source, target, static_fac, lam, num_home_factors, num_season_factors, random_seed, train_percentage, cost = sys.argv[
                                                                                                             1:]
name = "{}-{}-{}-{}-{}-{}-{}-{}-{}".format(source, target, static_fac, lam, num_home_factors, num_season_factors,
                                           random_seed, train_percentage, cost)
directory = os.path.expanduser('~/aaai2017/transfer_{}_{}_{}/'.format(source, target, cost))

filename = os.path.join(directory, name + '.pkl')
pr = pickle.load(open(filename, 'r'))
out = {'static': static_fac, 'lam': lam, 'num_home_factors': num_home_factors,
       'num_season_factors': num_season_factors, 'train_percentage': int(train_percentage),
       'random_seed': random_seed}
appliance='hvac'
region='SanDiego'
year=2014
pred_df = pr[appliance][range(4, 10)]
from sklearn.metrics import mean_squared_error
appliance_df = create_matrix_region_appliance_year(region, year, appliance)

if appliance == "hvac":
    start, stop = 5, 11
else:
    start, stop = 1, 13
pred_df = pred_df.copy()
pred_df.columns = [['%s_%d' % (appliance, month) for month in range(start, stop)]]
gt_df = appliance_df[pred_df.columns].ix[pred_df.index]

aggregate_df = appliance_df.loc[pred_df.index][['aggregate_%d' % month for month in range(start, stop)]]

aggregate_df.columns = gt_df.columns
rows, cols = np.where((aggregate_df < 100))
for r, c in zip(rows, cols):
    r_i, c_i = aggregate_df.index[r], aggregate_df.columns[c]
    aggregate_df.loc[r_i, c_i] = np.NaN

gt_fraction = gt_df.div(aggregate_df) * 100
pred_fraction = pred_df.div(aggregate_df) * 100

gt_fraction_dropna = gt_fraction.unstack().dropna()
pred_fraction_dropna = pred_fraction.unstack().dropna()
index_intersection = gt_fraction_dropna.index.intersection(pred_fraction_dropna.index)
gt_fraction_dropna = gt_fraction_dropna.ix[index_intersection]
pred_fraction_dropna = pred_fraction_dropna.ix[index_intersection]
difference_error = (gt_fraction_dropna-pred_fraction_dropna).abs()

rms = np.sqrt(mean_squared_error(gt_fraction_dropna, pred_fraction_dropna))
"""
for appliance in APPLIANCES_ORDER[1:]:
	if appliance == "hvac":
		out[appliance] = compute_rmse_fraction(appliance, pr[appliance][range(4, 10)], 'SanDiego')[2]
	else:
		out[appliance] = compute_rmse_fraction(appliance, pr[appliance], 'SanDiego')[2]

"""

db = client.test_nipun
posts = db.posts
posts.insert_one(out)
