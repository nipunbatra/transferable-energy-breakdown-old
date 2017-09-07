from create_matrix import create_matrix_region_appliance_year
from subprocess import Popen
import os
# region = "Austin"
# year = 2014
import delegator

# Enter your username on the cluster
username = 'yj9xs'

# Location of .out and .err files
SLURM_OUT = "~/git/slurm_out"

# Create the SLURM out directory if it does not exist
if not os.path.exists(SLURM_OUT):
	os.makedirs(SLURM_OUT)

# Max. num running processes you want. This is to prevent hogging the cluster
MAX_NUM_MY_JOBS = 360
# Delay between jobs when we exceed the max. number of jobs we want on the cluster
DELAY_NUM_JOBS_EXCEEDED = 10
import time


for setting in ['normal', 'transfer']:
	for case in [4, 2]:
		for constant_use in ['True', 'False']:
			for static_use in ['True', 'False']:
				for source in ['Austin']:
					for target in ['SanDiego']:
						for random_seed in range(10):
							for train_percentage in [0]:
								OFILE = "{}/{}-{}-{}-{}-{}-{}-{}-{}.out".format(SLURM_OUT, setting, case, constant_use, static_use, source, target, random_seed, train_percentage)
								EFILE = "{}/{}-{}-{}-{}-{}-{}-{}-{}.err".format(SLURM_OUT, setting, case, constant_use, static_use, source, target, random_seed, train_percentage)
								SLURM_SCRIPT = "{}/{}-{}-{}-{}-{}-{}-{}-{}.pbs".format('pbs_files', setting, case, constant_use, static_use, source, target, random_seed, train_percentage)
								CMD = 'python graph_laplacian_parallel.py {} {} {} {} {} {} {} {}'.format(setting, case, constant_use, static_use, source, target, random_seed, train_percentage)
								lines = []
								lines.append("#!/bin/sh\n")
								lines.append('#SBATCH --time=1-16:0:00\n')
								lines.append('#SBATCH --mem=16\n')
								lines.append('#SBATCH -c 24')
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