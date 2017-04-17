import numpy as np
import pandas as pd
from scipy.optimize import nnls
np.random.seed(0)


def version_rms_loss(A, K=2, cost_function='rms', numiter=1000):
	M, N = A.shape
	W = np.abs(np.random.uniform(low=0, high=1, size=(M, K)))
	H = np.abs(np.random.uniform(low=0, high=1, size=(K, N)))
	W = np.divide(W, K*W.max())
	H = np.divide(H, K*H.max())
	if cost_function=='rms':
		cost = cost_rms
	elif cost_function=='percentage':
		cost = cost_rms()

	for i in range(numiter):
		if i % 2 == 0:
			# Learn H, given A and W
			for j in range(N):
				mask_rows = pd.Series(A[:, j]).notnull()
				H[:, j] = nnls(W[mask_rows], A[:, j][mask_rows])[0]
		else:
			for j in range(M):
				mask_rows = pd.Series(A[j, :]).notnull()
				W[j, :] = nnls(H.transpose()[mask_rows], A[j, :][mask_rows])[0]
	return W, H

def cost_rms(A, W, H):
    from numpy import linalg
    mask = pd.DataFrame(A).notnull().values
    WH = np.dot(W, H)
    WH_mask = WH[mask]
    A_mask = A[mask]
    A_WH_mask = A_mask-WH_mask
    # Since now A_WH_mask is a vector, we use L2 instead of Frobenius norm for matrix
    return linalg.norm(A_WH_mask, 2)

def cost_percentage_loss(A, W, H):
    from numpy import linalg
    mask = pd.DataFrame(A).notnull().values
    WH = np.dot(W, H)
    WH_mask = WH[mask]
    A_mask = A[mask]
    A_WH_mask = A_mask-WH_mask
    abs_A_WH_mask = np.abs(A_WH_mask)
    percentage_error = np.divide(abs_A_WH_mask, A_mask)
    # Since now A_WH_mask is a vector, we use L2 instead of Frobenius norm for matrix
    return linalg.norm(percentage_error, 2)


num_iter = 1000
num_display_cost = max(int(num_iter/10), 1)
