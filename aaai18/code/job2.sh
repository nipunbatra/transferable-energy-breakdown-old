#!bin/bash

for i in 4
do
	for j in 6 8 9 10 15 20 30 40 50 60 70 80 90 100
	do
		python graph_laplacian_parallel_static.py normal 2 True True SanDiego Austin $i $j 1 13
	done
done
