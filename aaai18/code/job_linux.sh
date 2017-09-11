#!bin/bash

#python save_A_graph_laplacian_all.py 2 Asutin False
for random_seed in 5 6 7 8 9
do
	for train_percentage in 6 7 8 9 10 15 20 30 40 50 60 70 80 90 100
	do
		python graph_laplacian_parallel.py normal 2 True False Austin SanDiego $random_seed $train_percentage
		python graph_laplacian_parallel.py normal 2 False True Austin SanDiego $random_seed $train_percentage
		python graph_laplacian_parallel.py transfer 2 True False Austin SanDiego $random_seed $train_percentage
		python graph_laplacian_parallel.py transfer 2 False True Austin SanDiego $random_seed $train_percentage
	done
done