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
MAX_NUM_MY_JOBS = 140
# Delay between jobs when we exceed the max. number of jobs we want on the cluster
DELAY_NUM_JOBS_EXCEEDED = 10
import time

for static_fac in ['None','static']:
	for lam in [0.001, 0.01, 0.1, 1]:
		for num_home_factors in range(3, 9):
			for num_season_factors in range(1, 9):
				for random_seed in range(5):
					for train_percentage in range(10, 110, 10):


						OFILE = "{}/{}-{}-{}-{}-{}-{}.out".format(SLURM_OUT, static_fac, lam, num_home_factors, num_season_factors, random_seed, train_percentage )
						EFILE = "{}/{}-{}-{}-{}-{}-{}.err".format(SLURM_OUT, static_fac, lam, num_home_factors, num_season_factors, random_seed, train_percentage )
						SLURM_SCRIPT = "{}-{}-{}-{}-{}-{}.pbs".format(static_fac, lam, num_home_factors, num_season_factors, random_seed, train_percentage)
						CMD = 'python sparse-transfer.py {} {} {} {} {} {}'.format(static_fac, lam, num_home_factors, num_season_factors, random_seed, train_percentage)
						lines = []
						lines.append("#!/bin/sh\n")
						lines.append('#SBATCH --time=1-16:0:00\n')
						lines.append('#SBATCH --mem=16\n')
						lines.append('#SBATCH -o ' + '"' + OFILE + '"\n')
						lines.append('#SBATCH -e ' + '"' + EFILE + '"\n')
						lines.append(CMD + '\n')
						with open(SLURM_SCRIPT, 'w') as f:
							f.writelines(lines)
						command = ['sbatch', SLURM_SCRIPT]
						while len(delegator.run('squeue -u %s' % username).out.split("\n")) > MAX_NUM_MY_JOBS + 2:
							time.sleep(DELAY_NUM_JOBS_EXCEEDED)

						delegator.run(command, block=False)
						print SLURM_SCRIPT
