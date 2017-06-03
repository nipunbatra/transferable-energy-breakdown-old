region = "Austin"
year = 2014
APPLIANCES = ['fridge', 'hvac', 'wm', 'mw', 'oven', 'dw']
SLURM_OUT = "../../slurm_out"
import time
from subprocess import Popen

for features in ['energy', 'energy_static']:
	if features=="energy":
		min_f = 1
	else:
		# If using static features, should have >3 latent factors
		min_f = 3
	for cost in ['absolute']:
		for all_features in ['False']:
			for latent_factors in range(min_f, 12):
				OFILE = "%s/%s_%s_%s_%d.out" % (
					SLURM_OUT, cost[0], all_features[0], features[-1], latent_factors)
				EFILE = "%s/%s_%s_%s_%d.err" % (
					SLURM_OUT, cost[0], all_features[0], features[-1], latent_factors)
				SLURM_SCRIPT = "%s_%s_%s_%d.pbs" % (
					cost[0], all_features[0], features[-1], latent_factors)
				CMD = 'python mf_all_appliances_harness.py %s %s %s %d' % (
					features, cost, all_features, latent_factors)
				lines = []
				lines.append("#!/bin/sh\n")
				lines.append('#SBATCH --time=2-36:0:00\n')
				lines.append('#SBATCH --mem=16\n')
				lines.append('#SBATCH -o ' + '"' + OFILE + '"\n')
				lines.append('#SBATCH -e ' + '"' + EFILE + '"\n')
				lines.append(CMD + '\n')

				with open(SLURM_SCRIPT, 'w') as f:
					f.writelines(lines)
				command = ['sbatch', SLURM_SCRIPT]
				Popen(command)
				print SLURM_SCRIPT
