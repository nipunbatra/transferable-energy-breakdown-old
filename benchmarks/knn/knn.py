import sys
sys.path.append('../../code/')

from common_functions import *
ALL_REGIONS = ['Austin','SanDiego','Boulder']
ALL_FRACTION = {k:1.0 for k in ALL_REGIONS}

APPLIANCES = ['fridge','hvac']
test_region = 'SanDiego'

test_region_list = ['Austin','SanDiego','Boulder']
train_regions_list = [['Austin'],['Boulder'],['SanDiego'],
['Austin','SanDiego'],['Austin','Boulder'],['Boulder','SanDiego'],['Austin','Boulder','SanDiego']]

k = 2
feature_list=['energy','home']
year = 2014
out = {}
for appliance in APPLIANCES:
    print appliance
    out[appliance] = {}
    for train_regions in train_regions_list:
        out[appliance]['_'.join(train_regions)] = {}
        for test_region in test_region_list:
            print appliance, train_regions, test_region
            print "*"*40
            train_fraction_dict = {k:1.0 for k in train_regions}
            feature_list=['energy','home']
            year = 2014
            test_home = 54


            df_main, dfc_main = create_df_main(appliance, year, train_regions, train_fraction_dict,
                    test_region, test_home, feature_list)
            valid_homes = valid_homes_data[test_region][appliance]

            if appliance=="hvac":
                start, stop = 5, 11
            else:
                start, stop = 1, 13
            out_small = {}
            for test_home in valid_homes[:2]:


                df_main, dfc_main = create_df_main(appliance, year, train_regions, train_fraction_dict,
                                test_region, test_home, feature_list)
                df_main_norm = df_main.div(df_main.max())


                if appliance=="hvac":
                    start, stop = 5, 11
                else:
                    start, stop = 1, 13

                df = df_main_norm.copy()

                appliance_cols = ['%s_%d' %(appliance, month) for month in range(start, stop)]
                aggregate_columns = ['aggregate_%d' %month for month in range(1, 13)]

                feature_columns = [aggregate_columns]
                if 'home' in feature_list:
                    feature_columns.append(['area','total_occupants'])

                features = flatten(feature_columns)


                df_features = df[features]
                df_appliance = df_main[appliance_cols]
                df_test_home = df_main.ix[test_home][appliance_cols]

                o = {}
                valid_data_test_home = df_test_home.dropna().index
                for m in valid_data_test_home:
                    if m in valid_data_test_home:
                        valid_train_homes = df_appliance[m].dropna().index
                        df_train = df_features.ix[valid_train_homes]

                        # Find correlation
                        corr = df_train.T.corr()
                        corr_test_home = corr.ix[test_home].drop(test_home)
                        top_k_nghbrs = corr_test_home.sort_values().tail(k).index
                        o[m] = df_appliance[m].ix[top_k_nghbrs].mean()
                        #print top_k_nghbrs, m, test_home

                    else:
                        o[m] = np.NaN
                out[test_home] = pd.Series(o)
            pred_df = pd.DataFrame(out).T
            all_cols = np.intersect1d(appliance_cols,pred_df.columns )
            pred_df = pred_df[all_cols]
            df_mains,temp = create_df_main(appliance, year, ALL_REGIONS, ALL_FRACTION,
                            test_region, test_home, feature_list)
            gt_df = df_mains.ix[pred_df.index][all_cols]
            aggregate_columns = ['aggregate_%d' %month for month in range(start, stop)]
            aggregate_df = df_mains.ix[gt_df.index][aggregate_columns]
            aggregate_df.columns = gt_df.columns

            pred_fraction = pred_df.div(aggregate_df)
            gt_fraction = gt_df.div(aggregate_df)
            error_fraction = (pred_fraction-gt_fraction).abs().div(gt_fraction).mul(100)
            out[appliance]['_'.join(train_regions)][test_region] = error_fraction.unstack().mean()
            print out[appliance]['_'.join(train_regions)][test_region]
