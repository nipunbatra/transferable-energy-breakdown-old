from create_matrix import create_matrix_region_appliance_year
from subprocess import Popen

region = "Austin"
year = 2014
APPLIANCES = ['fridge', 'hvac', 'wm', 'mw', 'oven', 'dw']
SLURM_OUT = "../../slurm_out"
import time

for appliance in APPLIANCES[:]:
	appliance_df = create_matrix_region_appliance_year(region, year, appliance)
	for cost in ['abs', 'rel']:
		for all_features in ['True', 'False']:
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
						time.sleep(1)
						Popen(command)
						print SLURM_SCRIPT
