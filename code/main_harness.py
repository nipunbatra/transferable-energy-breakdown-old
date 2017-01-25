APPLIANCES = ['hvac','fridge']
TRAIN_REGIONS = ['Austin','Boulder','SanDiego']
TRAIN_FRACTIONS = {'Austin':0.0,'Boulder':0.0,'SanDiego':0}
TEST_REGION = 'SanDiego'

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
df = create_region_df(region, 2014)

df_copy = df.copy()
#drop_rows_having_no_data
o = {}
for h in df.index:
    o[h]=len(df.ix[h][feature_map['Monthly+Static']].dropna())
num_features_ser = pd.Series(o)
drop_rows = num_features_ser[num_features_ser==0].index

df = df.drop(drop_rows)
dfc = df.copy()






for appliance in APPLIANCES:

    if appliance=="hvac":
        start, end = 5,11
    else:
        start, end=1,13
    homes_appliance_region =valid_homes_data[region][appliance]

    for austin_fraction in np.linspace(0.0,1.0,11):
        for boulder_fraction in np.linspace(0.0,1.0,11):
            for sd_fraction in np.linspace(0.0,1.0,11):



                for home in homes_appliance_region:
                    OFILE = "%s/%s_%d.out" % (SLURM_OUT, appliance, home)
                    EFILE = "%s/%s_%d.err" % (SLURM_OUT, appliance, home)
                    SLURM_SCRIPT = "%s_%d.pbs" %(appliance, home)
                    CMD = 'python both_regions_dd_test_region_curve.py %s %d %d %d %d' %(appliance, home, home_var, case, num_homes)
                    lines = []
                    lines.append("#!/bin/sh\n")
                    lines.append('#SBATCH --time=0-01:0:00\n')
                    lines.append('#SBATCH --mem=16\n')
                    lines.append('#SBATCH -o '+'"' +OFILE+'"\n')
                    lines.append('#SBATCH -e '+'"' +EFILE+'"\n')
                    lines.append(CMD+'\n')

                with open(SLURM_SCRIPT, 'w') as f:
                   f.writelines(lines)
                command = ['sbatch', SLURM_SCRIPT]
                time.sleep(1)
                Popen(command)
                print CMD
        print "Now sleeping..."
