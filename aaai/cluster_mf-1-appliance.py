region = "SanDiego"
year = 2014
APPLIANCES = ['fridge', 'hvac', 'wm', 'mw', 'oven', 'dw']
SLURM_OUT = "../../slurm_out"
import time
from subprocess import Popen

for appliance in APPLIANCES:
	for features in ['energy', 'energy_static']:

		for cost in ['absolute']:
			for random_seed in range(5):
				for train_percentage in range(10, 110, 10):
					OFILE = "%s/%s_%s_%s_%s_%d.out" % (
						SLURM_OUT, appliance[0], cost[0], features[-1], random_seed, train_percentage)
					EFILE = "%s/%s_%s_%s_%s_%d.err" % (
						SLURM_OUT, appliance[0], cost[0], features[-1], random_seed, train_percentage)
					SLURM_SCRIPT = "%s_%s_%s_%d_%d.pbs" % (
						appliance[0], cost[0], features[-1], random_seed, train_percentage)
					CMD = 'python mf_1_appliances_harness.py %s %s %s %d %d' % (
						appliance, features, cost, random_seed, train_percentage)
					lines = []
					lines.append("#!/bin/sh\n")
					lines.append('#SBATCH --time=0-36:0:00\n')
					lines.append('#SBATCH --mem=16\n')
					lines.append('#SBATCH -o ' + '"' + OFILE + '"\n')
					lines.append('#SBATCH -e ' + '"' + EFILE + '"\n')
					lines.append(CMD + '\n')

					with open(SLURM_SCRIPT, 'w') as f:
						f.writelines(lines)
					command = ['sbatch', SLURM_SCRIPT]
					time.sleep(1)
					Popen(command)
					print SLURM_SCRIPT
