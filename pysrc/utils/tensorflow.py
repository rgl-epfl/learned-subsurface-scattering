import tensorflow as tf
import numpy as np


def py_func(func, inp, Tout, stateful=True, name=None, grad=None):
    """Wraps pyFunc to support custom gradients for pyfunc"""
    # Need to generate a unique name to avoid duplicates:
    rnd_name = 'PyFuncGrad' + str(np.random.randint(0, 1E+8))

    tf.RegisterGradient(rnd_name)(grad)  # see _MySquareGrad for grad example
    g = tf.get_default_graph()
    with g.gradient_override_map({"PyFunc": rnd_name}):
        return tf.py_func(func, inp, Tout, stateful=stateful, name=name)


def weight_variable(shape):
    # TODO: Potentially can also pass seed here if global seed is not enough
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))


def bias_variable(shape):
    return tf.Variable(tf.constant(0.05, shape=shape))


def conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME'):
    return tf.nn.conv2d(x, W, strides=strides, padding=padding)


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                          padding='SAME')


def conv2d_layer(x, input_channels, output_channels, kernel_size, activation_fn=tf.nn.relu, strides=[1, 1, 1, 1], padding='SAME'):
    W = weight_variable([kernel_size, kernel_size, input_channels, output_channels])
    b = bias_variable([output_channels])
    if activation_fn is not None:
        h = activation_fn(conv2d(x, W, strides) + b)
    else:
        h = conv2d(x, W, strides, padding=padding) + b
    return h

def conv2d_layer_new(x, output_channels, kernel_size, name=None, activation_fn=tf.nn.relu):
    return tf.layers.conv2d(x, output_channels, kernel_size, name=name, activation=activation_fn, 
                            kernel_initializer=None, 
                            bias_initializer=tf.constant_initializer(0.05), 
                            trainable=True)

def conv3d(x, W, strides=[1, 1, 1, 1, 1]):
    return tf.nn.conv3d(x, W, strides=strides, padding='SAME')


def max_pool_2x2x2(x):
    return tf.nn.max_pool3d(x, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1],
                            padding='SAME')


def conv3d_layer(x, input_channels, output_channels, kernel_size, activation_fn=tf.nn.relu, strides=[1, 1, 1, 1, 1]):
    W = weight_variable([kernel_size, kernel_size, kernel_size, input_channels, output_channels])
    b = bias_variable([output_channels])

    if activation_fn is not None:
        h = activation_fn(conv3d(x, W, strides) + b)
    else:
        h = conv3d(x, W, strides) + b
    return h


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def floats_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
