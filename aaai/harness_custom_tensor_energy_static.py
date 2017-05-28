from create_matrix import create_matrix_region_appliance_year
from subprocess import Popen
import os

region = "Austin"
year = 2014
APPLIANCES = ['fridge', 'hvac', 'wm', 'mw', 'oven', 'dw']
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

for appliance in APPLIANCES[:]:
	appliance_df = create_matrix_region_appliance_year(region, year, appliance)
	for all_features in ['False']:
	#for all_features in ['True', 'False']:
		#for cost in ['abs', 'rel']:
		for cost in ['abs']:


			for t in ['None', 'weather']:
				for h in ['None', 'static']:
					for a in range(1, 10):
						OFILE = "%s/%s_%s_%s_%d_%s_%s.out" % (SLURM_OUT, appliance[0], cost[0], all_features[0], a, h[0], t[0])
						EFILE = "%s/%s_%s_%s_%d_%s_%s.err" % (SLURM_OUT, appliance[0], cost[0], all_features[0], a, h[0], t[0])
						SLURM_SCRIPT = "%s_%s_%s_%d_%s_%s.pbs" % (appliance[0], cost[0], all_features[0],  a, h[0], t[0])
						CMD = 'python tensor_fact_custom_case_2_static_region_harness.py %s %s %d %s %s %s' % (
							appliance, all_features, a, h, t, cost)
						lines = []
						lines.append("#!/bin/sh\n")
						lines.append('#SBATCH --time=0-32:0:00\n')
						lines.append('#SBATCH --mem=16\n')
						lines.append('#SBATCH -o ' + '"' + OFILE + '"\n')
						lines.append('#SBATCH -e ' + '"' + EFILE + '"\n')
						lines.append(CMD + '\n')

						with open(SLURM_SCRIPT, 'w') as f:
							f.writelines(lines)
						command = ['sbatch', SLURM_SCRIPT]
						# Check our running processes to be less than max., else sleep
						while len(delegator.run('squeue -u %s' % username).out.split("\n")) > MAX_NUM_MY_JOBS + 2:
							time.sleep(DELAY_NUM_JOBS_EXCEEDED)

						delegator.run(command, block=False)
						print SLURM_SCRIPT
