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


for method in ['transfer', 'normal']:
	for cost in ['abs', 'rel']:
		for iterations in [2000, 10000, 20000]:
			OFILE = "%s/%s_%s_%d_0.1.out" % (SLURM_OUT, method, cost, iterations)
			EFILE = "%s/%s_%s_%d_0.1.err" % (SLURM_OUT, method, cost, iterations)
			SLURM_SCRIPT = "%s_%s_%d_0.1.pbs" % (method, cost, iterations)
			CMD = 'python normal_transfer_cluster_apart.py %s %s %d' % (method, cost, iterations)
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
