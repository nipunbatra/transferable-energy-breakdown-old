#!bin/bash

python save_A_graph_laplacian_all.py 4 Asutin True
for random_seed in {1..5}
do
	for train_percentage in 6 7 8 9 10 15 20 30 40 50 60 70 80 90 100
	do
		python graph_laplacian_parallel.py normal 4 True True Austin SanDiego $random_seed $train_percentage
		python graph_laplacian_parallel.py normal 4 True False Austin SanDiego $random_seed $train_percentage
		python graph_laplacian_parallel.py transfer 4 True True Austin SanDiego $random_seed $train_percentage
		python graph_laplacian_parallel.py transfer 4 True False Austin SanDiego $random_seed $train_percentage
	done
done
