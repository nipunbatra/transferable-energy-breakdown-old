import sys
from test_homes import valid_homes_data
ALL_HOMES, case = sys.argv[1:]
ALL_HOMES = bool(int(ALL_HOMES))
case = int(case)


if not ALL_HOMES:
    home_var = 0
else:
    home_var=1

print ALL_HOMES
SLURM_OUT = "../../slurm_out"
from subprocess import Popen
import time
from matrix_factorisation import nmf_features, transform, transform_2, preprocess, get_static_features
import  os

import numpy as np
import pandas as pd
import pickle
import sys
from subprocess import Popen
from features import feature_map
import subprocess


out_overall = pickle.load(open('../create_dataset/metadata/all_regions_years.pkl', 'r'))

region = "SanDiego"

df = out_overall[2014][region]

df_copy = df.copy()
#drop_rows_having_no_data
o = {}
for h in df.index:
    o[h]=len(df.ix[h][feature_map['Monthly+Static']].dropna())
num_features_ser = pd.Series(o)
drop_rows = num_features_ser[num_features_ser==0].index

df = df.drop(drop_rows)
dfc = df.copy()


df = df.rename(columns={'house_num_rooms':'num_rooms',
                        'num_occupants':'total_occupants',
                        'difference_ratio_min_max':'ratio_difference_min_max'})

if not ALL_HOMES:
    df = df[(df.full_agg_available == 1) & (df.md_available == 1)]
    dfc = dfc.ix[df.index]




import time

#for appliance in ['hvac','fridge','dw','wm','mw','oven']:
for appliance in ['hvac','fridge']:

    if appliance=="hvac":
        start, end = 5,11
    else:
        start, end=1,13
    homes_appliance_region =valid_homes_data[region][appliance]
    total_num_homes = len(homes_appliance_region)
    for num_homes in range(4, total_num_homes, 4):
        print "*" * 40
        print num_homes, appliance
        print "*"*40

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
