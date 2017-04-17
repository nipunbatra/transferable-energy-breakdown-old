"""
Sample Usage
run main.py --year=2014 --appliance='fridge' --Austin_fraction=0.1 --SanDiego_fraction=0.0 --Boulder_fraction=0.0 --test_region="SanDiego" --test_home=26 --feature_list="energy, household, region" --seed=0 --cost="absolute" --useall=True


Usage: main.py --year=Y --appliance=A --Austin_fraction=au_frac --SanDiego_fraction=sd_frac --Boulder_fraction=bo_frac --test_region=TSR --test_home=TSH --feature_list=FL --seed=0 --cost=COST --useall=U

Options:
    --year=Y [2014, 2015, ..]
    --appliance=A  appliance [can be 'fridge','hvac', etc]
    --Austin_fraction=au_frac
    --SanDiego_fraction=sd_frac
    --Boulder_fraction=bo_frac
    --test_region=TSR test region [str]
    --test_home=TSH  test home [integer]
    --feature_list=FL Feature List [list]
    --seed = Seed Int
    --cost = Cost String
    --useall = UseAll Bool
"""

import sys, traceback

print sys.argv
from docopt import docopt
import pandas as pd
import time
import numpy as np
import os

np.random.seed(0)

from common_functions import create_region_df, features_dict, create_feature_combinations, create_df_main
from matrix_factorisation import nmf_features, prepare_df_factorisation, prepare_known_features, \
	create_matrix_factorised, create_prediction

arguments = docopt(__doc__)
appliance = arguments['--appliance']
cost = arguments['--cost']
use_all = bool(arguments['--useall'])
year = int(arguments['--year'])
seed = int(arguments['--seed'])
test_home = int(arguments['--test_home'])
train_regions = ["Austin", "Boulder", "SanDiego"]
train_fraction_dict = {region: float(arguments['--%s_fraction' % region]) for region in train_regions}
test_region = arguments['--test_region']
feature_list = [x.strip() for x in arguments['--feature_list'].split(",")]
feature_list_path = "_".join(feature_list)
base_path = os.path.expanduser("~/transfer_subset_seed")


def create_directory_path(base_path, train_fraction_dict):
	train_regions_string = "_".join([str(int(100 * train_fraction_dict[x])) for x in train_regions])
	directory_path = os.path.join(base_path, test_region,
	                              feature_list_path, train_regions_string)
	if not os.path.exists(os.path.join(directory_path)):
		os.makedirs(directory_path)
	return directory_path


def create_file_store_path(base_path, appliance, seed, lat, feature_comb, test_home):
	file_path = "%s/%d_%s_%d_%s_%d.csv" % (base_path, seed, appliance, lat, '_'.join(feature_comb), test_home)
	print file_path
	return os.path.expanduser(file_path)


def _save_results(appliance, seed, lat, feature_comb, test_home, pred_df):
	csv_path = create_file_store_path(dir_path, appliance, seed, lat, feature_comb, test_home)
	pred_df.to_csv(csv_path)


dir_path = create_directory_path(base_path, train_fraction_dict)
feature_combinations = create_feature_combinations(feature_list, 2)

for feature_comb in np.array(feature_combinations)[:]:
	df, dfc, X_matrix, X_normalised, col_max, col_min, \
	appliance_cols, aggregate_cols, static_features, max_f = prepare_df_factorisation(appliance, year, train_regions,
	                                                                                  train_fraction_dict,
	                                                                                  test_region, [test_home],
	                                                                                  feature_list, seed, use_all)
	idx_user, data_user = prepare_known_features(feature_comb, static_features, X_normalised)

	for lat in range(1, 2):
		print lat

		try:
			csv_path = create_file_store_path(dir_path, appliance, seed, lat, feature_comb, test_home)
			if os.path.isfile(csv_path):
				print "skipping", csv_path
				pass
			A = create_matrix_factorised(appliance, [test_home], X_normalised)
			print A
			X, Y, res = nmf_features(A=A, k=lat, constant=0.01, regularisation=False,
			                         idx_user=idx_user, data_user=data_user,
			                         idx_item=None, data_item=None, MAX_ITERS=10)
			pred_df = create_prediction(test_home, X, Y, X_normalised, appliance,
			                            col_max, col_min, appliance_cols)
			print pred_df
			_save_results(appliance, seed, lat, feature_comb, test_home, pred_df)
		except Exception, e:
			print e
