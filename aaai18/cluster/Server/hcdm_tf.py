"""
Run all the code on HCDM

"""
source = "Austin"
target = "Boulder"
shell_script = "{}_{}.sh".format(source, target)
f = open(shell_script, 'w')
for case in [4]:
	for constant_use in ['True', 'False']:
		for static_use in ['True', 'False']:
			for setting in ['normal','transfer']:
				for train_percentage in [10, 20, 30, 40,
				                         50, 60, 70, 80, 90, 100]:
					for random_seed in range(5):
						CMD = 'python ../../code/graph_laplacian.py {} {} {} {} {} {} {} {}  &> ../../logs/{}_{}_{}_{}_{}_{}_{}-{}.out &\n'.format(setting, case, constant_use, static_use, source, target, random_seed, train_percentage,
					                                                                                                     setting, case,
					                                                                                                     constant_use,
					                                                                                                     static_use,
					                                                                                                     source,
					                                                                                                     target,
					                                                                                                     random_seed,
					                                                                                                     train_percentage)
						f.write(CMD)
f.close()
