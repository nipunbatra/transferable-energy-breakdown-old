from create_matrix import create_matrix_region_appliance_year
from subprocess import Popen

region = "Austin"
year = 2014
APPLIANCES = ['fridge', 'hvac', 'wm', 'mw', 'oven', 'dw']
SLURM_OUT = "../../slurm_out"
import time


for appliance in APPLIANCES[:1]:
	appliance_df = create_matrix_region_appliance_year(region, year, appliance)
	for cost in ['abs','rel']:
		for all_features in ['True','False']:

			for case in range(2, 3):
				for a in range(1, 2):
					for home in appliance_df.index:
						OFILE = "%s/%s_%d.out" % (SLURM_OUT, appliance, home)
						EFILE = "%s/%s_%d.err" % (SLURM_OUT, appliance, home)
						SLURM_SCRIPT = "%s_%d.pbs" % (appliance, home)
						CMD = 'python tensor_fact_custom_harness.py %s %d %d %s %s' % (appliance, case, a, cost, all_features)
						lines = []
						lines.append("#!/bin/sh\n")
						lines.append('#SBATCH --time=0-01:0:00\n')
						lines.append('#SBATCH --mem=16\n')
						lines.append('#SBATCH -o ' + '"' + OFILE + '"\n')
						lines.append('#SBATCH -e ' + '"' + EFILE + '"\n')
						lines.append(CMD + '\n')

						with open(SLURM_SCRIPT, 'w') as f:
							f.writelines(lines)
						command = ['sbatch', SLURM_SCRIPT]
						time.sleep(2)
						print Popen(command)
					print "Now sleeping.."