import sys
sys.path.append('../../code/')

from common_functions import *
ALL_REGIONS = ['Austin','SanDiego','Boulder']
ALL_FRACTION = {k:1.0 for k in ALL_REGIONS}

APPLIANCES = ['fridge','hvac']
APPLIANCES = ['fridge']
test_region = 'SanDiego'
feature_list = ['energy', 'home','region']


def _find_region(ng):
    o = {k:0 for k in ALL_REGIONS}
    for h in ng:
        for region in ALL_REGIONS:
            if h in ALL_HOMES_REGION[region]:
                o[region]+= 1
                break
    return o


#test_region_list = ['Austin','SanDiego','Boulder']
test_region_list = ['SanDiego']
train_regions_list = [['Austin'],['Boulder'],['SanDiego'],
['Austin','SanDiego'],['Austin','Boulder'],['Boulder','SanDiego'],['Austin','Boulder','SanDiego']]


train_regions_list = [['SanDiego'],['SanDiego','Austin']]
train_regions_list = [['SanDiego']]
train_regions_list = [['SanDiego','Austin']]


feature_list=['energy','home','region']
#feature_list = ['energy','home']
#feature_list = ['energy','home']
year = 2014
out = {}
out_nghbr = {}
for train_regions in train_regions_list:
    print train_regions
    out['_'.join(train_regions)] = {}
    out_nghbr['_'.join(train_regions)] = {}
    for appliance in APPLIANCES:
        out['_'.join(train_regions)][appliance] = {}
        out_nghbr['_'.join(train_regions)][appliance] = {}
        for k in range(1, 6):
            print k, train_regions
            train_fraction_dict = {k:1.0 for k in train_regions}
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
            out_small_n = {}
            for test_home in valid_homes[:]:
                #print test_home
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
                if 'region' in feature_list:
                    feature_columns.append(['dd_1','dd_2','dd_3','dd_4','dd_5','dd_6','dd_7','dd_8','dd_9','dd_10','dd_11','dd_12'])

                features = flatten(feature_columns)


                df_features = df[features]
                df_appliance = df_main[appliance_cols]
                df_test_home = df_main.ix[test_home][appliance_cols]

                o = {}
                o_n = {}
                valid_data_test_home_months = df_test_home.dropna().index
                for m in valid_data_test_home_months:

                    valid_train_homes = df_appliance[m].dropna().index
                    df_train = df_features.ix[valid_train_homes]

                    # Find correlation
                    corr = df_train.T.corr()
                    corr_test_home = corr.ix[test_home].drop(test_home)
                    top_k_nghbrs = corr_test_home.sort_values().tail(k).index
                    o[m] = df_appliance[m].ix[top_k_nghbrs].mean()
                    o_n[m] = _find_region(top_k_nghbrs)
                    #print top_k_nghbrs, m, test_home, o_n[m]
                for m in np.setdiff1d(appliance_cols, valid_data_test_home_months):
                    o[m] = np.NaN


                out_small[test_home] = pd.Series(o)
                out_small_n[test_home] = pd.Series(o_n)
            pred_df = pd.DataFrame(out_small).T
            all_cols = np.intersect1d(appliance_cols,pred_df.columns )
            pred_df = pred_df[all_cols]
            temp = compute_rmse_fraction(appliance, pred_df)
            out['_'.join(train_regions)][appliance][k] = temp
            print temp, k

