from create_matrix import create_matrix_region_appliance_year
from subprocess import Popen
import os
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
import datetime

import sys

cost = 'l21'
source_1, source_2, target = sys.argv[1:]
for static_fac in ['None','static']:
	for lam in [0]:

		for random_seed in range(5)[:]:
			for train_percentage in range(10, 110, 10)[:]:
				for outer_loop_iteration in range(0, 10)[:]:
					for num_iterations_cv in range(100, 1300, 200)[:]:
						for num_season_factors_cv in range(3, 9, 1)[:]:
							for num_home_factors_cv in range(2, 8, 1)[:]:
								for source_ratio in [0, 0.25, 0.5, 0.75, 1]:


									OFILE = "{}/{}-{}-{}-{}-{}-{}-{}-{}.out".format(SLURM_OUT, source_1, source_2, target, static_fac, lam, random_seed, train_percentage, cost,  outer_loop_iteration, source_ratio, num_iterations_cv, num_season_factors_cv, num_home_factors_cv )
									EFILE = "{}/{}-{}-{}-{}-{}-{}-{}-{}-{}.err".format(SLURM_OUT, source_1, source_2, target, static_fac, lam, random_seed, train_percentage, cost,  outer_loop_iteration, source_ratio, num_iterations_cv, num_season_factors_cv, num_home_factors_cv )
									SLURM_SCRIPT = "{}-{}-{}-{}-{}-{}-{}-{}-{}.pbs".format(source_1, source_2, target, static_fac, lam, random_seed, train_percentage, cost,  outer_loop_iteration, source_ratio, num_iterations_cv, num_season_factors_cv, num_home_factors_cv)
									CMD = 'python sparse-transfer-nested-cv-multiple-source.py {} {} {} {} {} {} {} {} {}'.format(source_1, source_2, target, static_fac, lam, random_seed, train_percentage, cost,  outer_loop_iteration, source_ratio, num_iterations_cv, num_season_factors_cv, num_home_factors_cv)
									lines = []
									lines.append("#!/bin/sh\n")
									lines.append('#SBATCH --time=02:0:00\n')
									lines.append('#SBATCH --mem=16\n')
									lines.append('#SBATCH -o ' + '"' + OFILE + '"\n')
									lines.append('#SBATCH -e ' + '"' + EFILE + '"\n')
									#lines.append('#SBATCH --exclude=artemis5\n')
									#lines.append('#SBATCH --exclude=artemis4\n')
									lines.append('#SBATCH --exclude=artemis3\n')
									lines.append(CMD + '\n')
									with open(SLURM_SCRIPT, 'w') as f:
										f.writelines(lines)
									command = ['sbatch', SLURM_SCRIPT]
									while len(delegator.run('squeue -u %s' % username).out.split("\n")) > MAX_NUM_MY_JOBS + 2:
										print(datetime.datetime.now(), "Waiting...")
										time.sleep(DELAY_NUM_JOBS_EXCEEDED)

									delegator.run(command, block=False)
									print SLURM_SCRIPT
