import tensorflow as tf


def deep():
    sess = tf.InteractiveSession()

    # The first two dimensions are the patch size,
    # the next is the number of input channels,
    # and the last is the number of output channels
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])


def weight_variable(shape):
    initial = tf.truncated_normal(shape=shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAMe')
