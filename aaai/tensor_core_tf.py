import tensorflow as tf
import numpy as np
import pandas as pd


def tensor_fact(E, h, t, cost='absolute', max_iter = 5000):
	np.random.seed(0)
	E_np = E.copy()
	M, N, S = E_np.shape
	E = tf.constant(E, dtype='float32')
	H = tf.Variable(tf.random_uniform([M, h], minval=0.1, maxval=1.0))
	A = tf.Variable(tf.random_uniform([h, N, t], minval=0.1, maxval=1.0))
	T = tf.Variable(tf.random_uniform([t, S], minval=0.1, maxval=1.0))

	HA = tf.tensordot(H, A, axes=[[1], [0]])
	HAT = tf.tensordot(HA, T, axes=[[2], [0]])

	mask = pd.Panel(E_np).notnull().values

	if cost == 'absolute':
		lr = 0.01
		f_norm = tf.reduce_sum(tf.pow(tf.boolean_mask(E, mask) - tf.boolean_mask(HAT, mask), 2))
	else:
		lr = 0.01
		nr = tf.boolean_mask(E, mask) - tf.boolean_mask(HAT, mask)
		dr = tf.maximum(tf.boolean_mask(E, mask) + 1e-4, tf.boolean_mask(HAT, mask))
		f_norm = tf.reduce_mean(tf.pow(nr/dr, 2))

	#lr = 0.01
	optimize = tf.train.AdagradOptimizer(lr).minimize(f_norm)



	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		for i in xrange(max_iter):
			loss, _ = sess.run([f_norm, optimize])

		A_out = sess.run(A)
		H_out = sess.run(H)
		T_out = sess.run(T)
		HAT_out = sess.run(HAT)
	return H_out, A_out, T_out
