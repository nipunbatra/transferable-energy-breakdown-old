import sys
import numpy as np
sys.path.append('../../code/')

from common_functions import *

APPLIANCES = ['fridge','hvac','wm']
ALL_REGIONS = ['Austin','SanDiego','Boulder']
ALL_FRACTION = {k:1.0 for k in ALL_REGIONS}
test_region_list = ['Austin','SanDiego','Boulder']
train_regions_list = [['Austin'],['Boulder'],['SanDiego'],
['Austin','SanDiego'],['Austin','Boulder'],['Boulder','SanDiego'],['Austin','Boulder','SanDiego']]

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
            for test_home in valid_homes[:]:
                print test_home
                df, dfc = create_df_main(appliance, year, train_regions, train_fraction_dict,
                                test_region, test_home, feature_list )

                appliance_cols = ['%s_%d' %(appliance, month) for month in range(start, stop)]
                df = df[appliance_cols]
                df = df[df.ix[test_home].dropna().index]
                #print df
                if len(df)>2:
                    #Closest index
                    if len((df-df.ix[test_home]).drop(test_home).dropna()):
                        # Only if there exist atleast one home having all the features
                        best_nghbr = (df-df.ix[test_home]).drop(test_home).dropna().apply(np.square).sum(axis=1).sort_values().index[0]
                        best_prediction = df.ix[best_nghbr]
                        out_small[test_home] = best_prediction
            pred_df = pd.DataFrame(out_small).T
            # Boulder doesnt have fridge
            all_cols = np.intersect1d(appliance_cols,pred_df.columns )
            pred_df = pred_df[all_cols]
            out_overall[appliance]= pred_df

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
