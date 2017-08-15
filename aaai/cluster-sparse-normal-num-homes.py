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
for static_fac in ['None','static']:
	for lam in [0]:

		for random_seed in range(5):
			for num_homes in range(4, 40, 8):


				OFILE = "{}/{}-{}-{}-{}.out".format(SLURM_OUT, static_fac[0], lam, random_seed, num_homes )
				EFILE = "{}/{}-{}-{}-{}.err".format(SLURM_OUT, static_fac[0], lam, random_seed, num_homes )
				SLURM_SCRIPT = "{}-{}-{}-{}.pbs".format(static_fac, lam, random_seed, num_homes)
				CMD = 'python sparse-normal-num-homes-cv.py {} {} {} {} {}'.format(source, static_fac, lam, random_seed, num_homes)
				lines = []
				lines.append("#!/bin/sh\n")
				lines.append('#SBATCH --time=1-16:0:00\n')
				lines.append('#SBATCH --mem=16\n')
				lines.append('#SBATCH -o ' + '"' + OFILE + '"\n')
				lines.append('#SBATCH -e ' + '"' + EFILE + '"\n')
				lines.append('#SBATCH --exclude=artemis5\n')
				lines.append('#SBATCH --exclude=artemis4\n')
				lines.append('#SBATCH --exclude=artemis3\n')
				lines.append(CMD + '\n')
				with open(SLURM_SCRIPT, 'w') as f:
					f.writelines(lines)
				command = ['sbatch', SLURM_SCRIPT]
				while len(delegator.run('squeue -u %s' % username).out.split("\n")) > MAX_NUM_MY_JOBS + 2:
					time.sleep(DELAY_NUM_JOBS_EXCEEDED)

				delegator.run(command, block=False)
				print SLURM_SCRIPT
