import tensorflow as tf
import numpy as np
import pandas as pd

np.random.seed(0)


def factorize(A_df, rank, lr=0.001, numiter=1000, cost_function='absolute'):
	"""
	
	:param A: pd.DataFrame
	:param rank: 
	:param numiter: 
	:return: 
	"""

	np_mask = A_df.notnull()

	# Boolean mask for computing cost only on valid (not missing) entries
	tf_mask = tf.Variable(np_mask.values)

	A = tf.constant(A_df.values)
	shape = A_df.values.shape

	# Initializing random H and W
	temp_H = np.random.randn(rank, shape[1]).astype(np.float32)
	temp_H = np.divide(temp_H, temp_H.max())

	temp_W = np.random.randn(shape[0], rank).astype(np.float32)
	temp_W = np.divide(temp_W, temp_W.max())

	H = tf.Variable(temp_H)
	W = tf.Variable(temp_W)
	WH = tf.matmul(W, H)

	abs_error = tf.boolean_mask(A, tf_mask) - tf.boolean_mask(WH, tf_mask)
	rel_error = tf.divide(abs_error, tf.boolean_mask(A, tf_mask))

	if cost_function == "absolute":
		cost = tf.reduce_sum(tf.pow(abs_error, 2))
	else:
		cost = tf.reduce_sum(tf.pow(rel_error, 2))

	train_step = tf.train.GradientDescentOptimizer(lr).minimize(cost)
	init = tf.global_variables_initializer()

	# Clipping operation. This ensure that W and H learnt are non-negative
	clip_W = W.assign(tf.maximum(tf.zeros_like(W), W))
	clip_H = H.assign(tf.maximum(tf.zeros_like(H), H))
	clip = tf.group(clip_W, clip_H)

	with tf.Session() as sess:
		sess.run(init)
		for i in range(numiter):
			sess.run(train_step)
			sess.run(clip)

		learnt_W = pd.DataFrame(sess.run(W))
		learnt_H = pd.DataFrame(sess.run(H))

	return learnt_W, learnt_H
