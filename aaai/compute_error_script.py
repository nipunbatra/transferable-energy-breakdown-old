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

for method  in ['normal', 'transfer']:
	for algo in ['adagrad', 'gd', 'gd_decay']:
		for cost in ['abs', 'rel']:
			for a in [2, 3]:
				for lr in [0.01, 0.1, 1, 10, 100]:
					OFILE = "%s/out_%s_%s_%s_%d_%f.out" % (SLURM_OUT, method, algo, cost, a, lr)
					EFILE = "%s/out_%s_%s_%s_%d_%f.err" % (SLURM_OUT, method, algo, cost, a, lr)
					SLURM_SCRIPT = "%s/out_%s_%s_%s_%d_%f.pbs" % ("./pbs_file", method, algo, cost, a, lr)
					CMD = 'python compute_error.py %s %s %s %d %f' % (method, algo, cost, a, lr)
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
