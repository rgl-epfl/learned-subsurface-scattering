
import numpy as np

import tensorflow as tf 

import models.nn

def coupling_layer(x, layer_name, layer_index, is_training, n_layers, layer_width, parameters=None,
                   inverse=False, use_first_layer_feats=False, net_type='mlp', 
                    use_coord_offset=False, dim=3):
    
    with tf.variable_scope(layer_name, reuse=inverse):
        n_channels = x.shape[1]
        split = n_channels // 2
        if use_coord_offset:
            n_channels = dim
            split = n_channels // 2
            mask1 = (np.arange(n_channels) + layer_index) % n_channels < split
            mask = np.arange(n_channels)[mask1]
            not_mask = np.arange(n_channels)[np.logical_not(mask1)]
            x_unstacked = tf.unstack(x, axis=1)
            xA = tf.stack([x_unstacked[t] for t in mask], axis=1)
            xB = tf.stack([x_unstacked[t] for t in not_mask], axis=1)
        else:
            odd = layer_index % 2 == 0
            if odd:
                xA = x[:, :split]
                xB = x[:, split:]
            else:
                xA = x[:, split:]
                xB = x[:, :split]

        # 1. Pass through the first part
        yA = xA

        # 2. Remap second part
        if net_type == 'mlp':
            net_fun = models.nn.multilayer_fcn 
        elif net_type == 'resnet':
            net_fun = models.nn.multilayer_resnet
        elif net_type == 'legacyresnet':
            net_fun = models.nn.multilayer_resnet_legacy

        if use_first_layer_feats:
            net_result = net_fun(tf.concat([xA, parameters], axis=1), is_training, None, 
                                 n_layers, layer_width, use_batch_norm=False, reuse=inverse, name='m_net')
        else:
            net_result = net_fun(xA, is_training, parameters, n_layers, layer_width, use_batch_norm=False,
                                reuse=inverse, name='m_net')

        n_params = int(xB.shape[1])
        # TODO: These could maybe also be depending on parameters?
        s = tf.contrib.layers.fully_connected(net_result, n_params, scope='s_fun', reuse=inverse, activation_fn=None)
        t = tf.contrib.layers.fully_connected(net_result, n_params, scope='t_fun', reuse=inverse, activation_fn=None)

        # Rescale s before applying the exponential to prevent overflow issues
        scale = tf.get_variable("rescaling_scale", [], initializer=tf.constant_initializer(0.), trainable=True)
        s = scale * tf.tanh(s)

        if inverse:
            yB = (xB - t) * tf.exp(-s)
        else:
            yB = xB * tf.exp(s) + t

        # Recombine data into one vector
        if use_coord_offset:
            yA_unstacked = tf.unstack(yA, axis=1)
            yB_unstacked = tf.unstack(yB, axis=1)
            out = [0] * dim
            for i, m in enumerate(mask):
                out[m] = yA_unstacked[i]
            for i, m in enumerate(not_mask):
                out[m] = yB_unstacked[i]
            out = tf.stack(out, axis=1)
        else:
            if odd:
                out = tf.concat([yA, yB], axis=1)
            else:
                out = tf.concat([yB, yA], axis=1)

        if inverse:  # Also return the exponential term which is part of the Jacobian of the sampling strategy
            return out, tf.reduce_sum(s, axis=1)
        else:
            return out


def unit_cube_to_restricted_range(x, target_range):
    x = 2.0 * x  # [0, 2]
    x -= 1.0  # [-1, 1]
    x *= target_range  # [-.9, .9]
    x += 1.0  # [.1, 1.9]
    x /= 2.0 # [.05, .95]
    return x

def niceModel(data_x, sampled_u, uniform_prior, n_coupling_layers, n_nn_layers, layer_width,
              is_training, parameters=None, use_first_layer_feats=False, net_type='mlp',
              pss_range=0.9, use_coord_offset=False, dim=3):
    with tf.variable_scope('nice'):
        # Build forward mapping (for sampling)
        sampled_x = sampled_u  # in [0, 1]^n
        if uniform_prior:
            sampled_x = unit_cube_to_restricted_range(sampled_x, pss_range)
            sampled_x = models.nn.logit(sampled_x)  # [0, 1]^n -> R^n

        for i in range(n_coupling_layers):
            sampled_x = coupling_layer(sampled_x, f'coupling{i}', i, is_training, n_nn_layers, layer_width, 
                                    parameters, False, use_first_layer_feats, net_type, use_coord_offset, dim=dim)


        if uniform_prior:
            sampled_x = tf.sigmoid(sampled_x)  # R^n -> [0, 1]^n

        # Backward mapping and Jacobian evaluation (for training and PDF evaluation)
        log_jacobian = tf.zeros([tf.shape(data_x)[0], 1])
        rec_u = data_x
        if uniform_prior:
            rec_u = unit_cube_to_restricted_range(rec_u, pss_range)
            rec_u = models.nn.logit(rec_u)  # [0, 1]^n -> R^n
            log_jacobian += tf.reduce_sum(-rec_u - 2 * tf.log(tf.exp(-rec_u) + 1), axis=1)

        for i in reversed(range(n_coupling_layers)):
            rec_u, log_det_sum = coupling_layer(
                rec_u, f'coupling{i}', i, is_training, 
                n_nn_layers, layer_width, parameters, True, use_first_layer_feats,
                 net_type, use_coord_offset, dim=dim)
            log_jacobian += log_det_sum[:, tf.newaxis]

        if uniform_prior:
            rec_u = tf.sigmoid(rec_u)  # R^n -> [0, 1]^n
            log_jacobian += tf.reduce_sum(-tf.log(rec_u - tf.square(rec_u)), axis=1)
        return sampled_x, rec_u, log_jacobian
