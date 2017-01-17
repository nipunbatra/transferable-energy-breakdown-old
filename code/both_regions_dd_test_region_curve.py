###################CASES#######################################
### CASE 1: Train and test on homes from SD w/o static features
### CASE 2: Train on homes from SD and Austin w/o static features
### Case 3: Train on homes from SD and Austin w/ static and temp feature
### Case 4: Train and test on homes from SD w static and temp features
### Case 5: Train on homes from
################################################################


from matrix_factorisation import nmf_features, transform, transform_2, \
    preprocess, get_static_features, get_static_features_region_level

from common_functions import create_region_df
import  os

import numpy as np
import pandas as pd
import pickle
import sys
sys.path.append("../../code")
from features import feature_map


base_path =os.path.expanduser("~/transfer")

def create_directory_path(base_path, num_homes):
    directory_path = os.path.join(base_path, str(num_homes))
    if not os.path.exists(os.path.join(directory_path)):
        os.makedirs(directory_path)

def create_file_store_path(base_path, num_homes, case, appliance, lat, feature_comb, test_home):
    return os.path.expanduser("%s/%d/%d_%s_%d_%s_%d.csv" %(base_path, num_homes, case, appliance, lat, '_'.join(feature_comb), test_home))

def _save_results(num_homes, case, appliance, lat, feature_comb, test_home, pred_df):
    create_directory_path(base_path, num_homes)
    csv_path = create_file_store_path(base_path, num_homes, case, appliance, lat, feature_comb, test_home)
    pred_df.to_csv(csv_path)





dds = {'Austin':[x/747.0 for x in [0, 16, 97, 292, 438, 579, 724, 747, 617, 376, 122, 46]],
       'SanDiego':[x/747.0 for x in [67, 92, 219, 183, 135, 272, 392, 478, 524, 451, 118, 32]]}

dd_keys = ['dd_'+str(x) for x in range(1,13)]


appliance, test_home, ALL_HOMES, case, num_homes_test_region = sys.argv[1:]
test_home = int(test_home)
ALL_HOMES =bool(int(ALL_HOMES))
case = int(case)
num_homes_test_region = int(num_homes_test_region)

aus_df, aus_dfc = create_region_df("Austin")
sd_df, sd_dfc = create_region_df("SanDiego")

from test_homes import valid_homes_data
austin_valid_homes = valid_homes_data['Austin'][appliance]
sd_valid_homes = valid_homes_data['SanDiego'][appliance]

aus_df = aus_df.ix[austin_valid_homes]
aus_dfc = aus_dfc.ix[austin_valid_homes]

sd_df = sd_df.ix[sd_valid_homes]
sd_dfc = sd_dfc.ix[sd_valid_homes]

all_homes_but_test = np.setdiff1d(sd_df.index.values, test_home)

sd_df = pd.concat([sd_df.ix[[test_home]], sd_df.ix[all_homes_but_test].head(num_homes_test_region)])
sd_dfc = pd.concat([sd_dfc.ix[[test_home]], sd_dfc.ix[all_homes_but_test].head(num_homes_test_region)])



if case==1:
    df = sd_df
    dfc = sd_dfc
elif case==2:
    df = pd.concat([sd_df, aus_df])
    dfc = pd.concat([sd_dfc, aus_dfc])
elif case==3:
    for key_num, key in enumerate(dd_keys):
        sd_df[key]=dds['SanDiego'][key_num]
        sd_dfc[key]=dds['SanDiego'][key_num]
        aus_df[key]=dds['Austin'][key_num]
        aus_dfc[key]=dds['Austin'][key_num]
    df = pd.concat([sd_df, aus_df])
    dfc = pd.concat([sd_dfc, aus_dfc])
elif case==4:
    df = sd_df
    dfc = sd_dfc
    """
    for key_num, key in enumerate(dd_keys):
        sd_df[key]=dds['SanDiego'][key_num]
        sd_dfc[key]=dds['SanDiego'][key_num]
    """







if not ALL_HOMES:
    df = df[(df.full_agg_available == 1) & (df.md_available == 1)]
    dfc = dfc.ix[df.index]


import itertools
feature_combinations = [['None']]
for l in range(1,2):
    for a in itertools.combinations(['occ','area','rooms','dd_1','dd_2','dd_3','dd_4','dd_5',
                                     'dd_6','dd_7','dd_8','dd_9','dd_10',
                                     'dd_11','dd_12'
                                     ], l):
        feature_combinations.append(list(a))



out = {}

if appliance=="hvac":
    start, end = 5,11
else:
    start, end=1,13
X_matrix, X_normalised, col_max, col_min, appliance_cols, aggregate_cols = preprocess(df, dfc, appliance)

if case>=3:
    static_features = get_static_features_region_level(dfc, X_normalised)
else:
    static_features = get_static_features(dfc, X_normalised)


from copy import deepcopy
all_cols = deepcopy(appliance_cols)
all_cols.extend(aggregate_cols)
#all_feature_homes = dfc[(dfc.full_agg_available == 1) & (dfc.md_available == 1)][all_cols].dropna().index


if case==3:
    max_f = 20
else:
    max_f=3

for feature_comb in np.array(feature_combinations)[:max_f]:
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

    for lat in range(1,6):
        try:
            print lat

            # Check if exists or not
            csv_path = create_file_store_path(base_path, num_homes_test_region, case, appliance, lat, feature_comb, test_home)
            if os.path.isfile(csv_path):
                print "skipping",csv_path
                pass
            else:

                if lat<len(feature_comb):
                    continue
                out[tuple(feature_comb)][lat]={}

                X_home = X_normalised.copy()
                for month in range(start, end):
                    X_home.loc[test_home, '%s_%d' %(appliance, month)] = np.NAN
                mask = X_home.notnull().values
                # Ensure repeatably random problem data.
                A = X_home.copy()
                X, Y, res = nmf_features(A, lat, 0.01, False, idx_user, data_user, 7)

                pred_df = pd.DataFrame(Y*X)
                pred_df.columns = X_normalised.columns
                pred_df.index = X_normalised.index
                out[tuple(feature_comb)][lat] = transform_2(pred_df.ix[test_home], appliance, col_max, col_min)[appliance_cols]
                pred_df = transform_2(pred_df.ix[test_home], appliance, col_max, col_min)[appliance_cols]
                _save_results(num_homes_test_region, case, appliance, lat, feature_comb, test_home, pred_df)
        except Exception, e:
            print "Exception occurred", e
