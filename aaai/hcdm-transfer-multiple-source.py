from create_matrix import create_matrix_region_appliance_year
from subprocess import Popen
import os
# region = "Austin"
# year = 2014
import delegator

# Enter your username on the cluster
username = 'nb2cz'

# Location of .out and .err files
SLURM_OUT = "../../slurm_out"

# Create the SLURM out directory if it does not exist
if not os.path.exists(SLURM_OUT):
	os.makedirs(SLURM_OUT)

# Max. num running processes you want. This is to prevent hogging the cluster
MAX_NUM_MY_JOBS = 150
# Delay between jobs when we exceed the max. number of jobs we want on the cluster
DELAY_NUM_JOBS_EXCEEDED = 10
import time

source_1 = 'SanDiego'
source_2 = 'Boulder'
target = 'Austin'
cost = 'l21'
for static_fac in ['None']:
	for lam in [0]:

		for random_seed in range(5):
			for train_percentage in range(10, 110, 10):

					CMD = 'python sparse-transfer-cv-multiple-source.py {} {} {} {} {} {} {} &> {}_{}_{}_{}.out &'.format(source_1, source_2, target, static_fac, lam, random_seed, train_percentage, source_1, source_2, train_percentage, random_seed)
					print CMD
