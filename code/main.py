"""
Sample Usage
run main.py --year=2014 --appliance='fridge' --Austin_fraction=0.1 --SanDiego_fraction=0.0 --Boulder_fraction=0.0 --test_region="SanDiego" --test_home=26 --feature_list="energy, household, region"


Usage: main.py --year=Y --appliance=A --Austin_fraction=au_frac --SanDiego_fraction=sd_frac --Boulder_fraction=bo_frac --test_region=TSR --test_home=TSH --feature_list=FL

Options:
    --year=Y [2014, 2015, ..]
    --appliance=A  appliance [can be 'fridge','hvac', etc]
    --Austin_fraction=au_frac
    --SanDiego_fraction=sd_frac
    --Boulder_fraction=bo_frac
    --test_region=TSR test region [str]
    --test_home=TSH  test home [integer]
    --feature_list=FL Feature List [list]
"""

import sys, traceback
print sys.argv
from docopt import docopt
import pandas as pd
import time
import numpy as np
import os

from common_functions import create_region_df, features_dict, create_feature_combinations
from test_homes import valid_homes_data
from degree_days import dds, dd_keys
from matrix_factorisation import nmf_features, transform, transform_2, \
    preprocess, get_static_features, get_static_features_region_level



arguments = docopt(__doc__)
appliance=arguments['--appliance']

year = int(arguments['--year'])
test_home = int(arguments['--test_home'])
train_regions = ["Austin","Boulder","SanDiego"]
train_fraction_dict = {region:float(arguments['--%s_fraction' %region]) for region in train_regions}
test_region=arguments['--test_region']
feature_list = [x.strip() for x in arguments['--feature_list'].split(",")]
feature_list_path = "_".join(feature_list)
base_path =os.path.expanduser("~/transfer")

def create_directory_path(base_path, train_fraction_dict):
    train_regions_string = "_".join([str(int(100*train_fraction_dict[x])) for x in train_regions])
    directory_path = os.path.join(base_path, test_region,
                                  feature_list_path,train_regions_string)
    if not os.path.exists(os.path.join(directory_path)):
        os.makedirs(directory_path)
    return directory_path

def create_file_store_path(base_path, appliance, lat, feature_comb, test_home):
    return os.path.expanduser("%s/%s_%d_%s_%d.csv" %(base_path, appliance, lat, '_'.join(feature_comb), test_home))

def _save_results(appliance, lat, feature_comb, test_home, pred_df):
    csv_path = create_file_store_path(dir_path, appliance, lat, feature_comb, test_home)
    pred_df.to_csv(csv_path)


dir_path = create_directory_path(base_path, train_fraction_dict)

dfs = {}
dfcs = {}

for train_region in train_regions:
    temp_df, temp_dfc = create_region_df(train_region, year)
    temp_valid_homes = valid_homes_data[train_region][appliance]
    temp_df = temp_df.ix[temp_valid_homes]
    # Number of homes to use from this region
    temp_num_homes = int(len(temp_dfc)*train_fraction_dict[train_region])
    # Choosing subset of homes
    temp_df = temp_df.head(temp_num_homes)
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

out = {}

if appliance=="hvac":
    start, end = 5,11
else:
    start, end=1,13

end_df_prep = time.time()

start_norm = time.time()
X_matrix, X_normalised, col_max, col_min, appliance_cols, aggregate_cols = preprocess(df, dfc, appliance)
end_norm = time.time()
print "Normalisation took", end_norm-start_norm


start_features = time.time()
if "region" in feature_list:
    static_features = get_static_features_region_level(dfc, X_normalised)
else:
    static_features = get_static_features(dfc, X_normalised)
end_features = time.time()
print "Static features took", end_features-start_features


if "region" in feature_list:
    max_f = 20
else:
    max_f=3

feature_combinations = create_feature_combinations(feature_list, 2)

for feature_comb in np.array(feature_combinations)[:]:
    start_misc =time.time()
    print feature_comb
    out[tuple(feature_comb)]={}
    if 'None' in feature_comb:
        idx_user=None
        data_user=None
    else:
        idx_user = {}
        data_user = {}
        dictionary_static = {}
        for feature in feature_comb:
            dictionary_static[feature]=static_features[feature]
        static_features_df = pd.DataFrame(dictionary_static, index=range(len(X_normalised.index)))
        for fe in static_features_df.columns:
            idx_user[fe]=np.where(static_features_df[fe].notnull())[0]
            data_user[fe]=static_features_df[fe].dropna().values
    end_misc = time.time()
    print "MISC took", end_misc-start_misc
    for lat in range(1,10):
        try:
            print lat

            # Check if exists or not
            csv_path = create_file_store_path(dir_path, appliance, lat, feature_comb, test_home)
            if os.path.isfile(csv_path):
                print "skipping",csv_path
                pass
            else:
                if lat<len(feature_comb):
                    continue
                out[tuple(feature_comb)][lat]={}
                start_pre = time.time()
                X_home = X_normalised.copy()
                for month in range(start, end):
                    X_home.loc[test_home, '%s_%d' %(appliance, month)] = np.NAN
                mask = X_home.notnull().values
                # Ensure repeatably random problem data.
                A = X_home.copy()
                end_pre = time.time()
                print "Preparing for NMF took", end_pre-start_pre
                start_mf = time.time()
                X, Y, res = nmf_features(A, lat, 0.01, False, idx_user, data_user, 10)
                end_mf = time.time()
                print "NMF took", end_mf-start_mf
                start_post = time.time()
                pred_df = pd.DataFrame(Y*X)
                pred_df.columns = X_normalised.columns
                pred_df.index = X_normalised.index
                out[tuple(feature_comb)][lat] = transform_2(pred_df.ix[test_home], appliance, col_max, col_min)[appliance_cols]
                pred_df = transform_2(pred_df.ix[test_home], appliance, col_max, col_min)[appliance_cols]
                print pred_df
                _save_results(appliance, lat, feature_comb, test_home, pred_df)
                end_post = time.time()
                print "POST took", end_post-start_post
        except Exception, e:
            print "Exception occurred", e
            traceback.print_exc()
