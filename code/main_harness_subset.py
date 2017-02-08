APPLIANCES = ['hvac','fridge']
TRAIN_REGIONS = ['Austin','Boulder','SanDiego']
TEST_REGION = 'SanDiego'

FEATURE_LISTS = [
    'energy',
    'energy, home',
    'energy, region',
    'energy, home, region'
]



import sys
from test_homes import valid_homes_data

SLURM_OUT = "../../slurm_out"
from subprocess import Popen
import time
from matrix_factorisation import nmf_features, transform, transform_2, preprocess, get_static_features
from common_functions import create_region_df
import  os

import numpy as np
import pandas as pd
import pickle
import sys
from subprocess import Popen
from features import feature_map
import subprocess


region='SanDiego'
df, dfc = create_region_df(region, 2014)

df_copy = df.copy()
#drop_rows_having_no_data
o = {}
for h in df.index:
    o[h]=len(df.ix[h][feature_map['Monthly+Static']].dropna())
num_features_ser = pd.Series(o)
drop_rows = num_features_ser[num_features_ser==0].index

df = df.drop(drop_rows)
dfc = df.copy()





test_region_string = '--test_region=%s' %(TEST_REGION)
feature_string = '--feature_list='

for appliance in APPLIANCES:
    appliance_string = '--appliance=%s' % (appliance)

    if appliance=="hvac":
        start, end = 5,11
    else:
        start, end=1,13
    homes_appliance_region =valid_homes_data[region][appliance]

    for fe in FEATURE_LISTS:
        feature_string = "--feature_list='%s'" %fe
        #time.sleep(120)

        for austin_fraction in np.linspace(0.0,1.0,6):
            for boulder_fraction in np.linspace(0.0,1.0,6):
                for sd_fraction in np.linspace(0.0,1.0,6):
                    fraction_string = "--Austin_fraction=%.2f --SanDiego_fraction=%.2f --Boulder_fraction=%.2f" %(austin_fraction, sd_fraction, boulder_fraction)
                    for home in homes_appliance_region:
                        test_home_string = '--test_home=%d' %(home)

                        total_string = 'python main.py --year=2014'
                        total_string = " ".join([total_string, appliance_string,
                                                 fraction_string, test_region_string,
                                                 test_home_string, feature_string])
                        print total_string



                        OFILE = "%s/%s.out" % (SLURM_OUT, total_string)
                        EFILE = "%s/%s.err" % (SLURM_OUT, total_string)
                        SLURM_SCRIPT = "%s_%d_%s.pbs" %(appliance, home, fe)
                        lines = []
                        lines.append("#!/bin/sh\n")
                        lines.append('#SBATCH --time=0-01:0:00\n')
                        lines.append('#SBATCH --mem=16\n')
                        lines.append('#SBATCH -o '+'"' +OFILE+'"\n')
                        lines.append('#SBATCH -e '+'"' +EFILE+'"\n')
                        lines.append(total_string+'\n')

                        with open(SLURM_SCRIPT, 'w') as f:
                           f.writelines(lines)
                        command = ['sbatch', SLURM_SCRIPT]
                        time.sleep(0.5)
                        Popen(command)
        time.sleep(120)
    #time.sleep(600)
