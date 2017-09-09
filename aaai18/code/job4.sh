#!bin/bash

#python save_A_graph_laplacian_all.py 4 Asutin False
for random_seed in {0..4}
do
	for train_percentage in 6 7 8 9 10 15 20 30 40 50 60 70 80 90 100
	do
		#python graph_laplacian_parallel_austin.py normal 4 True False SanDiego Austin $random_seed $train_percentage 1 13
		python graph_laplacian_parallel_austin.py normal 4 False False SanDiego Austin $random_seed $train_percentage 1 13
		python graph_laplacian_parallel_austin.py normal 4 False True SanDiego Austin $random_seed $train_percentage 1 13
		#python graph_laplacian_parallel_austin.py normal 4 True True SanDiego Austin $random_seed $train_percentage 1 13
	done
done
