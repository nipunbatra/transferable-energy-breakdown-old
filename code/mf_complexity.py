"""
Program to measure the time complexity of MF wrt

1. #Number of rows(m)
2. #Number of columns(n)
3. Rank (r)
"""

# Create dummy non-negative m,n matrix

import pandas as pd
import numpy as np
from matrix_factorisation import  nmf_features
import time

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




