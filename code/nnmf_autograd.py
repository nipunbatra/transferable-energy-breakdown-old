import autograd.numpy as np
import pandas as pd
from autograd import grad, multigrad


def factorize(A_df, rank, lr=0.01, numiter=5000, cost_function='absolute'):
	np.random.seed(0)
	A = A_df.values

	def cost_abs(W, H):
		pred = np.dot(W, H)
		mask = ~np.isnan(A)
		return np.sqrt(((pred - A)[mask].flatten() ** 2).mean(axis=None))

	def cost_rel(W, H):
		pred = np.dot(W, H)
		mask = ~np.isnan(A)
		abs_error = (pred - A)[mask].flatten()
		rel_error = np.divide(abs_error, A[mask].flatten())
		return np.sqrt((rel_error ** 2).mean(axis=None))

	shape = A_df.shape
	H = np.abs(np.random.randn(rank, shape[1]))
	H = np.divide(H, H.max())
	W = np.abs(np.random.randn(shape[0], rank))
	W = np.divide(W, W.max())

	learning_rate = lr

	if cost_function == "absolute":
		cost = cost_abs
	elif cost_function == "relative":
		cost = cost_rel

	grad_cost = multigrad(cost, argnums=[0, 1])
	for i in range(numiter):

		del_W, del_H = grad_cost(W, H)
		W = W - del_W * learning_rate
		H = H - del_H * learning_rate

		# Ensuring that W, H remain non-negative. This is also called projected gradient descent
		W[W < 0] = 0
		H[H < 0] = 0

	return W, H
