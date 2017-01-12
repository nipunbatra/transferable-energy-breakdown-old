import os, glob
import pandas as pd
path = os.path.expanduser('~/collab_all_homes_both_regions/')
#path = os.path.expanduser('~/subset_105/')

def compute_prediction_case(case, appliance, feature, k):
    files = glob.glob(path+'%d_%s_%d_%s_*.csv' %(case, appliance, k, feature))
    out = {}
    for e in files:
        out[int(e.split('_')[-1][:-4])] = pd.read_csv(e,index_col=0, header=None).squeeze()
    return pd.DataFrame(out).T

def compute_prediction(appliance, feature, k):
    files = glob.glob(path+'%s_%d_%s_*.csv' %(appliance, k, feature))
    out = {}
    for e in files:
        out[int(e.split('_')[-1][:-4])] = pd.read_csv(e,index_col=0, header=None).squeeze()
    return pd.DataFrame(out).T

def compute_prediction_subset(appliance, feature, latent_factors, ran, num_homes):
    files = glob.glob(path +'%d_%d_%s_%d_%s_*.csv' % (ran, num_homes, appliance, latent_factors, feature))
    out = {}
    for e in files:
        out[int(e.split('_')[-1][:-4])] = pd.read_csv(e,index_col=0, header=None).squeeze()
    return pd.DataFrame(out).T

def find_all_error():
    out = {}
    for appliance in ['wm','mw','oven','fridge','hvac','dw']:
        out[appliance]={}
        for feature in ['None', 'temperature','occ', 'area','rooms','occ_area','occ_rooms','area_rooms','occ_area_rooms']:
            out[appliance][feature]={}
            for k in range(1, 10):
                try:
                    print feature, k, appliance
                    pred_df = compute_prediction(appliance, feature, k)
                    gt_df = find_gt_df(appliance, pred_df)
                    out[appliance][feature][k] = find_error_df(gt_df, pred_df)
                except:
                    pass
    return out

def find_gt_df(appliance, pred_df):
    import pickle
    out_overall = pickle.load(open('/if6/nb2cz/git/Neighbourhood-NILM/data/input/all_regions.pkl', 'r'))
    region = "SanDiego"
    df = out_overall[region]
    gt_df = df[pred_df.columns].ix[pred_df.index]
    return gt_df


def find_error_df(gt_df, pred_df):
    return (pred_df-gt_df).abs().div(gt_df).mul(100)

def find_optimal(appliance):
    o = {}
    for feature in ['None','temperature', 'occ', 'area','rooms','occ_area','occ_rooms','area_rooms','occ_area_rooms']:
        o[feature]={}
        for k in range(1, 10):
            try:
                print feature, k
                pred_df = compute_prediction(appliance, feature, k)
                gt_df = find_gt_df(appliance, pred_df)
                o[feature][k] = find_error_df(gt_df, pred_df).median().mean()
                print o[feature][k], len(pred_df)
            except Exception, e:
                print e
    return pd.DataFrame(o)

def create_overall_dict():
    out = {}
    #for appliance in ['wm','mw','oven','fridge','hvac','dw']:
    for appliance in ['fridge']:
        out[appliance]={}
        for feature in ['None']:
        #for feature in ['None', 'temperature','occ', 'area','rooms','occ_area','occ_rooms','area_rooms','occ_area_rooms']:
            out[appliance][feature]={}
            for k in range(1, 10):
                try:
                    print feature, k, appliance
                    pred_df = compute_prediction(appliance, feature, k)

                    out[appliance][feature][k] = pred_df
                except:
                    pass
    return out


def create_overall_dict_case(case):
    out = {}
    #for appliance in ['wm','mw','oven','fridge','hvac','dw']:
    for appliance in ['fridge','hvac']:
        out[appliance]={}
        for feature in ['None','occ', 'area','rooms','dd_1',
                        'dd_2','dd_3','dd_4','dd_5','dd_6',
                        'dd_7','dd_8','dd_9','dd_10','dd_11','dd_12']:
        #for feature in ['None', 'temperature','occ', 'area','rooms','occ_area','occ_rooms','area_rooms','occ_area_rooms']:
            out[appliance][feature]={}
            for k in range(1, 10):
                try:
                    print feature, k, appliance
                    pred_df = compute_prediction_case(case, appliance, feature, k)

                    out[appliance][feature][k] = pred_df
                except:
                    pass
    return out

def create_overall_dict_subset():
    out = {}
    for num_homes in range(5, 55, 5):
        out[num_homes]={}
        #for appliance in ['wm','mw','oven','fridge','hvac','dw']:
        for appliance in ['hvac']:
            out[num_homes][appliance]={}
            for feature in ['None']:
            #for feature in ['None', 'occ', 'area','rooms','occ_area','occ_rooms','area_rooms','occ_area_rooms']:
                out[num_homes][appliance][feature]={}
                #for latent_factors in range(2, 10):
                for latent_factors in range(2,3):
                    out[num_homes][appliance][feature][latent_factors] = {}
                    for ran in range(10):
                        try:
                            print num_homes, feature, latent_factors, appliance
                            pred_df = compute_prediction_subset(appliance, feature, latent_factors, ran, num_homes)

                            out[num_homes][appliance][feature][latent_factors][ran] = pred_df
                        except Exception, e:
                            print e


