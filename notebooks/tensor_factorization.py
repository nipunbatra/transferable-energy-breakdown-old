
import tensorflow as tf
import numpy as np
import pandas as pd

def tf(E, h, t):
    E_np = E.copy()
    M, N, S = E_np.shape
    E = tf.constant(E, dtype='float32')

    initializer = tf.random_uniform_initializer(minval=0.2, maxval=1)
    H =  tf.get_variable("H", [M, h], initializer=initializer)
    A =  tf.get_variable(name="A", shape=[h, N, t], initializer=initializer)
    T =  tf.get_variable(name="T", shape=[t, S], initializer=initializer)

    HA = tf.tensordot(H, A,axes = [[1], [0]])
    HAT = tf.tensordot(HA, T, axes=[[2], [0]])

    mask = pd.Panel(E_np).notnull().values

    f_norm = tf.reduce_sum(tf.pow(tf.boolean_mask(E, mask) - tf.boolean_mask(HAT, mask), 2))

    lr = 0.01
    optimize = tf.train.AdagradOptimizer(lr).minimize(f_norm)

    max_iter=10000
    display_step = int(max_iter/10)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print "cost, #iterations"
        pre_loss = 10e12
        for i in xrange(max_iter):

            loss, _ = sess.run([f_norm, optimize])
            if i%display_step==0:
                print loss, ",", i
        A_out = sess.run(A)
        H_out = sess.run(H)
        T_out = sess.run(T)
        HAT_out = sess.run(HAT)
    return H, A, T
