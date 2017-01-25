import pickle
import pandas as pd
import numpy as np
import itertools
from features import feature_map

out_overall = pickle.load(open('../create_dataset/metadata/all_regions_years.pkl', 'r'))


def create_region_df(region, year=2014):
    df = out_overall[year][region]

    df_copy = df.copy()
    #drop_rows_having_no_data
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

features_dict = {}
features_dict['energy'] = ['None']
features_dict['region'] = ['dd_1','dd_2','dd_3','dd_4','dd_5','dd_6','dd_7','dd_8','dd_9','dd_10','dd_11','dd_12']
features_dict['home'] = ['occ','area','rooms']
