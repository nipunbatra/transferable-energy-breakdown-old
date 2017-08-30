"""
Run all the code on HCDM

"""
source = "Austin"
target = "SanDiego"
#shell_script = "{}_{}.sh".format(source, target)
#f = open(shell_script, 'w')
for case in [4]:
	#for constant_use in ['True', 'False']:
	for constant_use in ['False']:
		for static_use in ['False']:
			for setting in ['normal']:
			#for setting in ['normal','transfer']:
				for train_percentage in range(10, 110, 20):
					for random_seed in range(4):
						CMD = 'python ../../code/graph_laplacian.py {} {} {} {} {} {} {} {}  &> ../../logs/{}_{}_{}_{}_{}_{}_{}-{}.out &'.format(setting, case, constant_use, static_use, source, target, random_seed, train_percentage,
					                                                                                                     setting, case,
					                                                                                                     constant_use,
					                                                                                                     static_use,
					                                                                                                     source,
					                                                                                                     target,
					                                                                                                     random_seed,
					                                                                                                     train_percentage)
						print(CMD)
						#f.write(CMD)
f.close()
