from create_matrix import create_matrix_region_appliance_year
from subprocess import Popen
import os
# region = "Austin"
# year = 2014
import delegator

# Enter your username on the cluster
username = 'yj9xs'

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

for train_iter in [5000]:
	for algo in ['gd']:
		for static in ['static']:
			for lam in [0]:
				OFILE = "%s/out_%d_%s_%s_%f.out" % (SLURM_OUT, train_iter, algo, static, lam)
				EFILE = "%s/out_%d_%s_%s_%f.err" % (SLURM_OUT, train_iter, algo, static, lam)
				SLURM_SCRIPT = "%s/out_%d_%s_%s_%f.pbs" % ("./pbs_file", train_iter, algo, static, lam)
				CMD = 'python compute_error.py %d %s %s %f' % (train_iter, algo, static, lam)
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
