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

source = 'Austin'
target = 'Boulder'

for random_seed in range(4):
	for train_percentage in range(10, 110, 10):
		CMD = 'python graph_laplacian_transfer.py {} {} {} {} &> graph-lap-transfer-{}_{}_{}_{}.out &'.format(source, target,  random_seed, train_percentage, source, target, random_seed, train_percentage)
		print CMD

