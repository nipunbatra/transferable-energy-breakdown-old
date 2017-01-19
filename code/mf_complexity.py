"""
Program to measure the time complexity of MF wrt

1. #Number of rows(m)
2. #Number of columns(n)
3. Rank (r)
"""

# Create dummy non-negative m,n matrix

import pandas as pd
import numpy as np
from matrix_factorisation import nmf_features, preprocess
import time
from common_functions import create_region_df
from timeit import Timer, timeit, repeat



def random_data():
	m_max = 500
	n_max = 36

	data = np.abs(np.random.randn(n_max, m_max))
	df_max = pd.DataFrame(data)
	df_max = df_max.div(df_max.max())


	out = {}
	for m in range(10, 510, 10):
		out[m]={}
		for n in range(4, 40, 4):
			out[m][n]={}
			for rank in range(1, 10):

				df = df_max.head(m)[range(n)]

				start = time.time()
				a = nmf_features(df, rank, MAX_ITERS=10)
				end = time.time()
				out[m][n][rank] = end - start
				print m, n, rank, end-start


def actual_data(num_rows=100, num_cols=24, latent_features=2):
	df, dfc = create_region_df("Austin",2014)
	appliance="fridge"
	X_matrix, X_normalised, col_max, col_min, appliance_cols, aggregate_cols = preprocess(df, dfc, appliance)
	cols = X_normalised.columns[:num_cols]
	A = X_normalised.head(num_rows)[cols]
	X, Y, res = nmf_features(A, latent_features, 0.01, False, MAX_ITERS= 10)


def harness():
	out = {}
	for num_rows in range(50, 550, 50):
		out[num_rows] = {}
		for latent_factors in range(1, 10, 1):
			print num_rows, latent_factors
			statement = "actual_data(%d,24,%d)" % (num_rows, latent_factors)
			out[num_rows][latent_factors] = timeit(stmt=statement,
			                                       setup="from __main__ import actual_data",
			                                       number=5)
	return out
