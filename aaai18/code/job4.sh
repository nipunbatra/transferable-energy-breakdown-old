#!bin/bash

for i in 0 1 2 3 4
do
	for j in 6 8 9 10 15 20 30 40 50 60 70 80 90 100
	do
		echo $i $j
		python HBAS_parallel.py normal True Austin SanDiego $i $j 1 13
	done
done

#python graph_laplacian_parallel_static.py normal 2 True True SanDiego Austin 0 100 1 13
#python graph_laplacian_parallel_static.py normal 2 True False SanDiego Austin 0 90 1 13
#python graph_laplacian_parallel_static.py normal 2 True False SanDiego Austin 0 100 1 13


