import pickle
import pandas as pd
import numpy as np
import itertools
from features import feature_map
from degree_days import dds, dd_keys
import os
from sklearn.metrics import mean_squared_error

upper_limit = {
'hvac':40,
'fridge':10,
'wm':1
}

### FIX This
data_path = os.path.expanduser("~/git/scalable-nilm/create_dataset/metadata/all_regions_years.pkl")




out_overall = pickle.load(open(data_path, 'r'))


def compute_rmse_fraction(appliance, pred_df, region='Austin'):
    year=2014
    train_regions = [region]
    train_fraction_dict = {'SanDiego':1.0,'Austin':0.0,'Boulder':0.0}
    test_region=region
    test_home = pred_df.index[0]
    feature_list=['energy']
    df, dfc = create_df_main(appliance, year, train_regions, train_fraction_dict,
                    test_region, test_home, feature_list, seed=0)
    # pred_df = mf_pred[appliance][appliance_feature][latent_factors]
    gt_df = df[pred_df.columns].ix[pred_df.index]
    if appliance == "hvac":
        start, stop = 5, 11
    else:
        start, stop = 1, 13
    aggregate_df = df.ix[pred_df.index][['aggregate_%d' % month for month in range(start, stop)]]

    aggregate_df.columns = gt_df.columns
    rows, cols = np.where((aggregate_df < 100))
    for r, c in zip(rows, cols):
        r_i, c_i = aggregate_df.index[r], aggregate_df.columns[c]
        aggregate_df.loc[r_i, c_i] = np.NaN

    gt_fraction = gt_df.div(aggregate_df) * 100
    pred_fraction = pred_df.div(aggregate_df) * 100




    rms = np.sqrt(mean_squared_error(pred_fraction.unstack().dropna(), gt_fraction.unstack().dropna()))
    return rms


def create_region_df(region, year=2014):
    df = out_overall[year][region]

    df_copy = df.copy()
    # drop_rows_having_no_data
    o = {}
    for h in df.index:
        o[h]=len(df.ix[h][feature_map['Monthly']].dropna())
    num_features_ser = pd.Series(o)
    drop_rows = num_features_ser[num_features_ser==0].index

    df = df.drop(drop_rows)
    dfc = df.copy()


    df = df.rename(columns={'house_num_rooms':'num_rooms',
                            'num_occupants':'total_occupants',
                            'difference_ratio_min_max':'ratio_difference_min_max'})
    return df, dfc


def flatten(l):
    return [item for sublist in l for item in sublist]


def create_feature_combinations(list_of_features, size_combinations=2):
    f = []
    for fe in list_of_features:
        f.append(features_dict[fe])
    f = flatten(f)
    feature_comb = []
    for l in range(1, size_combinations+1):
        for a in itertools.combinations(f, l):
            feature_comb.append(list(a))
    return feature_comb


def create_df_main(appliance, year, train_regions, train_fraction_dict,
                test_region, test_home, feature_list, seed):
    dfs = {}
    dfcs = {}

    for train_region in train_regions:
        temp_df, temp_dfc = create_region_df(train_region, year)
        temp_valid_homes = valid_homes_data[train_region][appliance]
        temp_df = temp_df.ix[temp_valid_homes]
        # Choosing subset of homes
        if train_fraction_dict[train_region]<1.0:
            temp_df = temp_df.sample(frac=train_fraction_dict[train_region], random_state=seed)
        # Check that the test home is not in our data
        temp_df = temp_df.ix[[x for x in temp_df.index if x!=test_home]]
        temp_dfc = temp_dfc.ix[temp_df.index]
        # Add degree days
        if "region" in feature_list:
            for key_num, key in enumerate(dd_keys):
                temp_df[key]=dds[year][train_region][key_num]
                temp_dfc[key]=dds[year][train_region][key_num]
        dfs[train_region] = temp_df
        dfcs[train_region] = temp_dfc

    train_df = pd.concat(dfs.values())
    train_dfc = pd.concat(dfcs.values())


    test_df, test_dfc = create_region_df(test_region)
    test_df = test_df.ix[[test_home]]
    test_dfc = test_dfc.ix[[test_home]]

    if "region" in feature_list:
        for key_num, key in enumerate(dd_keys):
            test_df[key]=dds[year][test_region][key_num]
            test_dfc[key]=dds[year][test_region][key_num]

    df = pd.concat([train_df, test_df])
    dfc = pd.concat([train_dfc, test_dfc])

    if appliance=="hvac":
        start, stop = 5, 11
    else:
        start, stop = 1, 13

    appliance_cols = ['%s_%d' %(appliance, month) for month in range(start, stop)]
    df_a = df[appliance_cols]
    ul_appliance = upper_limit[appliance]

    rows, cols = np.where(df_a<ul_appliance)
    for i, j in zip(rows, cols):
        home = df_a.index[i]
        col = df_a.columns[j]
        df.loc[home, col] = np.NaN


    return df, dfc

features_dict = {}
features_dict['energy'] = ['None']
features_dict['region'] = ['dd_1','dd_2','dd_3','dd_4','dd_5','dd_6','dd_7','dd_8','dd_9','dd_10','dd_11','dd_12']
features_dict['home'] = ['occ','area','rooms']

from copy import deepcopy
import numpy as np

REGIONS = ['SanDiego','Austin','Boulder']
APPLIANCES = ['hvac','fridge','wm']

homes_with_problems = []

homes_with_problems = [
	8061, # Jan Aggregate data reading is wrong!
    3938, #HVAC usage is wrong
    2031, #HVAC wrong
    4083,
    4495,
    4761,
    4934,
    5938,
    6377,
    6547,
    8342,
    9775,
    9612,
    3864, #FRIDGE usage wrong
    7114, # HVAC 0
    8574
]

homes_with_problems = []


valid_homes_data = {}

ALL_HOMES_REGION = {}
for region in ['Austin','SanDiego','Boulder']:
    ALL_HOMES_REGION[region] = create_region_df(region)[0].index


def find_valid_homes(region, appliance, appliance_fraction, aggregate_fraction):
	df, dfc = create_region_df(region)
	if appliance == "hvac":
		start, stop = 5, 11
	else:
		start, stop = 1, 13
	appliance_cols = ['%s_%d' % (appliance, month) for month in range(start, stop)]
	aggregate_cols = ['aggregate_%d' % month for month in range(start, stop)]

	aggregate_num_cols, appliance_num_cols = len(aggregate_cols), len(appliance_cols)

	# Choose homes that have 1) atleast half the aggregate_cols and
	# 2) half the appliance cols

	# Condition #1
	valid_homes_aggregate = df.ix[df[aggregate_cols].notnull().sum(axis=1) >= aggregate_num_cols*aggregate_fraction].index

	# Condition #2
	valid_homes_appliance = df.ix[df[appliance_cols].notnull().sum(axis=1) >= appliance_num_cols*appliance_fraction].index

	# Homes meeting both
	valid_homes = np.intersect1d(valid_homes_aggregate, valid_homes_appliance)
	valid_homes = np.setdiff1d(valid_homes, homes_with_problems)
	return valid_homes


valid_homes_data = {}
for region in REGIONS:
	valid_homes_data[region] = {}
	for appliance in APPLIANCES:
		valid_homes_data[region][appliance] = find_valid_homes(region, appliance, 0.5, 0.5)
