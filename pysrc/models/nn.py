import os

import numpy as np
import tensorflow as tf
import time

import vae.utils
from utils.tensorflow import *
from vae.utils import *
from utils.mesh import sample_mesh
import vae.tf_utils


def logit(x):
    # return -tf.log(1 / x - 1)
    return tf.log(x) - tf.log(1. - x)


def dense(x, n_hidden, activation_fn=tf.nn.relu, use_batch_norm=False, phase=None, dense_layer=None, out_name=None):

    # h1 = tf.contrib.layers.fully_connected(x, n_hidden,
    #                                        activation_fn=None)

    if dense_layer is None:
        dense_layer = tf.layers.Dense(n_hidden, activation=None)
    x = dense_layer(x)
    if use_batch_norm:
        x = tf.contrib.layers.batch_norm(x, center=True, scale=True, is_training=phase)
    if activation_fn is None:
        return x, dense_layer
    return activation_fn(x, name=out_name), dense_layer


def handle_batch_norm(use_batch_norm, is_training, reuse, scope):
    if use_batch_norm:
        n_fn = tf.contrib.layers.batch_norm
        n_params = {'is_training': is_training, 'reuse': reuse, 'scope': scope}
    else:
        n_fn = None
        n_params = None
    return n_fn, n_params


def multilayer_fcn(x, is_training, parameters, n_layers, width, use_batch_norm, name, reuse=False,
                   use_dropout=False, dropout_keep_prob=None, regularizer=None, n_out=None):
    n_fn, n_params = handle_batch_norm(use_batch_norm, is_training, reuse, scope=name)

    def get_width(i):
        return width[i] if type(width) is list else width

    with tf.variable_scope(name, reuse=reuse):
        for i in range(n_layers):
            x = tf.contrib.layers.fully_connected(
                tf.concat([x, parameters], 1) if parameters is not None else x, get_width(i),
                scope=f'fcn_{i}', reuse=reuse, normalizer_fn=n_fn, normalizer_params=n_params,
                weights_regularizer=regularizer)

            if use_dropout and i < n_layers - 1:
                x = tf.nn.dropout(x, dropout_keep_prob)

        if n_out is not None:
            x = tf.contrib.layers.fully_connected(x, n_out, scope='fcn_out',
                reuse=reuse, weights_regularizer=regularizer, activation_fn=None)

    return x


def residual_block(x, width, regularizer, is_training, use_batch_norm, reuse=False):
    """https://arxiv.org/pdf/1603.05027.pdf"""
    x_old = x
    if use_batch_norm:
        x = tf.layers.batch_normalization(x, training=is_training, name='bn_0', reuse=reuse)
    x = tf.nn.relu(x)
    x = tf.contrib.layers.fully_connected(
        x, width, scope=f'fc_0', reuse=reuse, activation_fn=None, weights_regularizer=regularizer)

    if use_batch_norm:
        x = tf.layers.batch_normalization(x, training=is_training, name=f'bn_1', reuse=reuse)
    x = tf.nn.relu(x)
    x = tf.contrib.layers.fully_connected(
        x, width, scope=f'fc_1', reuse=reuse, activation_fn=None, weights_regularizer=regularizer)
    return x + x_old


def multilayer_resnet(input_layer, is_training, parameters, n_res_net_units, width,
                      use_batch_norm, name, reuse=False, regularizer=None):

    def get_width(i):  # TODO: Support funnel type architecture for resnets too
        return width[0] if type(width) is list else width

    with tf.variable_scope(name, reuse=reuse):
        x = tf.contrib.layers.fully_connected(
            tf.concat([input_layer, parameters], 1) if parameters is not None else input_layer,
            get_width(0), scope='resnet_00', reuse=reuse,
            activation_fn=None, weights_regularizer=regularizer)
        if use_batch_norm:
            x = tf.layers.batch_normalization(x, training=is_training, name='bn_00', reuse=reuse)
        for i in range(n_res_net_units):
            with tf.variable_scope(f'resnet_{i}', reuse=reuse):
                x = residual_block(x, get_width(i), regularizer, is_training, use_batch_norm, reuse)
        return tf.nn.relu(x)


def multilayer_resnet_legacy(input_layer, is_training, parameters, n_res_net_units, width,
                             use_batch_norm, name, reuse=False, regularizer=None):
    n_fn, n_params = handle_batch_norm(use_batch_norm, is_training, reuse, scope=name)

    def get_width(i):
        return width[i] if type(width) is list else width

    with tf.variable_scope(name, reuse=reuse):
        # Run one layer to get up to the used size of layers
        input_layer = tf.contrib.layers.fully_connected(
            tf.concat([input_layer, parameters], 1) if parameters is not None else input_layer,
            get_width(0), scope='resnet_00', reuse=reuse,
            normalizer_fn=n_fn, normalizer_params=n_params, weights_regularizer=regularizer)

        x = input_layer
        for i in range(n_res_net_units):
            x = tf.contrib.layers.fully_connected(
                tf.concat([x, parameters], 1) if parameters is not None else x, get_width(i), scope='resnet_{}_0'.format(i), reuse=reuse,
                normalizer_fn=n_fn, normalizer_params=n_params, weights_regularizer=regularizer)
            x = tf.contrib.layers.fully_connected(
                tf.concat([x, parameters], 1) if parameters is not None else x, get_width(i), scope='resnet_{}_1'.format(i),
                reuse=reuse, activation_fn=None, weights_regularizer=regularizer)

            x = x + input_layer
            if use_batch_norm:
                x = tf.contrib.layers.batch_norm(x, center=True, scale=True,
                                                 is_training=is_training, scope='resnet_bn_{}'.format(i), reuse=reuse)
            x = tf.nn.relu(x)
        return x


def create_conv_net_shallow(img):
    sizes = [1, 8, 16, 32]
    strides = [2, 2, 1]
    for i in range(3):
        img = conv2d_layer(img, sizes[i], sizes[i + 1], 3, tf.nn.relu, [1, strides[i], strides[i], 1])
        if strides[i] == 1:
            img = max_pool_2x2(img)
    return img


def create_conv_net_deep(img):
    sizes = [1, 16, 32, 64]
    strides = [2, 2, 1]
    for i in range(3):
        img = conv2d_layer(img, sizes[i], sizes[i + 1], 3, tf.nn.relu, [1, 1, 1, 1])
        img = conv2d_layer(img, sizes[i + 1], sizes[i + 1], 3, tf.nn.relu, [1, strides[i], strides[i], 1])
        if strides[i] == 1:
            img = max_pool_2x2(img)
    return img
