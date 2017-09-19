#!bin/bash

for i in 0 1 2 3 4 5 6 7 8 9
do
	python graph_laplacian_parallel_zero.py transfer 2 True True Austin SanDiego $i 0 1 13
	python graph_laplacian_parallel_zero.py transfer 2 True False Austin SanDiego $i 0 1 13
	python graph_laplacian_parallel_zero.py transfer 2 False False Austin SanDiego $i 0 1 13
	python graph_laplacian_parallel_zero.py transfer 2 False True Austin SanDiego $i 0 1 13
done
