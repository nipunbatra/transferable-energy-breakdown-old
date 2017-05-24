region = "Austin"
year = 2014
APPLIANCES = ['fridge', 'hvac', 'wm', 'mw', 'oven', 'dw']
SLURM_OUT = "../../slurm_out"
import time
from subprocess import Popen

for appliance in APPLIANCES:
	for features in ['energy', 'energy_static']:
		for cost in ['absolute', 'relative']:
			for all_features in ['True', 'False']:
				for latent_factors in range(1, 10):
					OFILE = "%s/%s_%s_%s_%s_%d.out" % (
						SLURM_OUT, appliance[0], cost[0], all_features[0], features[-1], latent_factors)
					EFILE = "%s/%s_%s_%s_%s_%d.err" % (
						SLURM_OUT, appliance[0], cost[0], all_features[0], features[-1], latent_factors)
					SLURM_SCRIPT = "%s_%s_%s_%d_%d.pbs" % (
						appliance[0], cost[0], all_features[0], features[-1], latent_factors)
					CMD = 'python mf_harness.py %s %s %s %s %d' % (
						appliance, features, cost, all_features, latent_factors)
					lines = []
					lines.append("#!/bin/sh\n")
					lines.append('#SBATCH --time=0-16:0:00\n')
					lines.append('#SBATCH --mem=16\n')
					lines.append('#SBATCH -o ' + '"' + OFILE + '"\n')
					lines.append('#SBATCH -e ' + '"' + EFILE + '"\n')
					lines.append(CMD + '\n')

					with open(SLURM_SCRIPT, 'w') as f:
						f.writelines(lines)
					command = ['sbatch', SLURM_SCRIPT]
					time.sleep(2)
					Popen(command)
					print SLURM_SCRIPT
