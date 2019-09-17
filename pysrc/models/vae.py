import tensorflow as tf

from models.nn import *

from vae.global_config import *


def mmd_loss(encoded_z, sampled_z):
    """Implements a maximum mean discrepancy loss estimate for a batch"""
    n = tf.cast(tf.shape(encoded_z)[0], tf.int32)
    nf = tf.cast(tf.shape(encoded_z)[0], tf.float32)

    norms_pz = tf.reduce_sum(tf.square(sampled_z), axis=1, keepdims=True)
    dotprods_pz = tf.matmul(sampled_z, sampled_z, transpose_b=True)
    distances_pz = norms_pz + tf.transpose(norms_pz) - 2.0 * dotprods_pz

    norms_qz = tf.reduce_sum(tf.square(encoded_z), axis=1, keepdims=True)
    dotprods_qz = tf.matmul(encoded_z, encoded_z, transpose_b=True)
    distances_qz = norms_qz + tf.transpose(norms_qz) - 2.0 * dotprods_qz

    dotprods = tf.matmul(encoded_z, sampled_z, transpose_b=True)

    distances = norms_qz + tf.transpose(norms_pz) - 2.0 * dotprods
    Cbase = 2.0 * tf.cast(tf.shape(encoded_z)[1], tf.float32)
    stat = 0.0
    for scale in [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]:
        C = scale * Cbase
        res1 = C / (C + distances_qz)
        res1 += C / (C + distances_pz)
        res1 = tf.multiply(res1, 1. - tf.eye(n))
        res1 = tf.reduce_sum(res1) / (nf * nf - nf)
        res2 = C / (C + distances)
        res2 = tf.reduce_sum(res2) * 2. / (nf * nf)
        stat += res1 - res2
    return stat


def encoder(x, is_training, parameters, config):
    with tf.name_scope('encoder'):
        if config.add_encoder_noise:
            # Particular instance of the implicit random encoder
            def add_noise(x):
                return x + tf.truncated_normal(tf.shape(x), 0.0, 0.01)

            def do_nothing(x):
                return x
            x = tf.cond(is_training, lambda: add_noise(x), lambda: do_nothing(x))

        if config.first_layer_feats:
            x = tf.concat([x, parameters], axis=1)
            parameters = None

        if config.use_res_net:
            n_res_net_units = int(np.ceil(config.n_mlp_layers / 2))
            x = multilayer_resnet(x, is_training, parameters, n_res_net_units,
                                  config.n_mlp_width, config.use_batch_norm, 'encoder_rn')
        else:
            x = multilayer_fcn(x, is_training, parameters, config.n_mlp_layers, config.n_mlp_width,
                               config.use_batch_norm, 'encoder_fcn')
        x = tf.concat([x, parameters], 1) if parameters is not None else x

        if config.use_wae_mmd:
            if config.wae_random_enc:
                # Constrain mean between -1 and 1
                z_mean, _ = dense(x, config.n_latent, activation_fn=None)
                # z_mean = tf.nn.sigmoid(z_mean) * 2.0 - 1.0

                z_log_sigma2, _ = dense(x, config.n_latent, activation_fn=None)
                samples = tf.random_normal([tf.shape(z_mean)[0], config.n_latent], 0, 1, dtype=tf.float32)
                z_encoded = z_mean + (tf.exp(z_log_sigma2 / 2) * samples)
                return z_encoded, z_mean, z_log_sigma2

            else:
                z_encoded, _ = dense(x, config.n_latent, activation_fn=None)
            return z_encoded
        else:
            z_mean, _ = dense(x, config.n_latent, activation_fn=None)
            z_log_sigma2, _ = dense(x, config.n_latent, activation_fn=None)
            return z_mean, z_log_sigma2


def decoder(n_outputs, z_mean, z_log_sigma2, is_training, sampled_z, parameters, config, estimated_z=None):
    with tf.name_scope('decoder'):

        if not config.use_wae_mmd:
            samples = tf.random_normal([tf.shape(z_mean)[0], config.n_latent], 0, 1, dtype=tf.float32)
            estimated_z = z_mean + (tf.exp(z_log_sigma2 / 2) * samples)

        if config.first_layer_feats:
            estimated_z = tf.concat([estimated_z, parameters], axis=1)
            sampled_z = tf.concat([sampled_z, parameters], axis=1)
            parameters = None

        if config.use_res_net:
            n_res_net_units = int(np.ceil(config.n_mlp_layers / 2))
            estimated_z = multilayer_resnet(estimated_z, is_training, parameters, n_res_net_units,
                                            config.n_mlp_width, config.use_batch_norm, 'decoder_rn')
            sampled_z = multilayer_resnet(sampled_z, False, parameters, n_res_net_units,
                                          config.n_mlp_width, config.use_batch_norm, 'decoder_rn', True)
        else:
            estimated_z = multilayer_fcn(estimated_z, is_training, parameters, config.n_mlp_layers,
                                         config.n_mlp_width, config.use_batch_norm, 'decoder_fcn')
            sampled_z = multilayer_fcn(sampled_z, False, parameters, config.n_mlp_layers,
                                       config.n_mlp_width, config.use_batch_norm, 'decoder_fcn', True)

        estimated_z = tf.concat([estimated_z, parameters], 1) if parameters is not None else estimated_z
        sampled_z = tf.concat([sampled_z, parameters], 1) if parameters is not None else sampled_z
        y, layer = dense(estimated_z, n_outputs, activation_fn=None)
        y2, _ = dense(sampled_z, n_outputs, activation_fn=None, dense_layer=layer)
        return y, y2


def projectiveDecoder(z, n_outputs, is_training, parameters, config, ph_manager):
    with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE):
        rate = 20.0
        for i in range(config.n_coupling_layers):
            if config.use_res_net:
                n_res_net_units = int(np.ceil(config.n_mlp_layers / 2))
                net_result = multilayer_resnet(z, is_training, parameters, n_res_net_units,
                                               config.n_mlp_width, config.use_batch_norm, 'decoder_rn', reuse=tf.AUTO_REUSE)
            else:
                net_result = multilayer_fcn(z, is_training, parameters, config.n_mlp_layers,
                                            config.n_mlp_width, config.use_batch_norm, 'decoder_fcn', reuse=tf.AUTO_REUSE)

            s = tf.contrib.layers.fully_connected(net_result, 3, scope='s_fun', activation_fn=None, reuse=tf.AUTO_REUSE)
            t = tf.contrib.layers.fully_connected(net_result, 3, scope='t_fun', activation_fn=None, reuse=tf.AUTO_REUSE)
            scale = tf.get_variable("rescaling_scale", [], initializer=tf.constant_initializer(0.), trainable=True)
            s = scale * tf.tanh(s)

            z = z * tf.exp(s) + t

            for i in range(1):
                value = vae.tf_utils.eval_poly(z, None, None, ph_manager.shape_features_p, config.poly_order(),
                                               tangent_space=False, scale=ph_manager.poly_scale_factor_p)
                grad = vae.tf_utils.tf_eval_poly_gradient(z, None, None, ph_manager.shape_features_p,
                                                          tangent_space=False, scale=ph_manager.poly_scale_factor_p, poly_order=config.poly_order())
                z = z - rate * grad * tf.sign(value)

        return z


def avb_encoder(x, sampled_z, is_training, parameters, config):
    with tf.name_scope('encoder'):
        x = tf.concat([x, sampled_z], axis=1)
        if config.use_res_net:
            n_res_net_units = int(np.ceil(config.n_mlp_layers / 2))
            x = multilayer_resnet(x, is_training, parameters, n_res_net_units,
                                  config.n_mlp_width, config.use_batch_norm, 'encoder_rn')
        else:
            x = multilayer_fcn(x, is_training, parameters, config.n_mlp_layers, config.n_mlp_width,
                               config.use_batch_norm, 'encoder_fcn')
        x = tf.concat([x, parameters], 1) if parameters is not None else x

        z, _ = dense(x, config.n_latent, activation_fn=None)
        return z


def avb_decoder(n_outputs, estimated_z, sampled_z, is_training, parameters, config):
    with tf.name_scope('decoder'):
        if config.use_res_net:
            n_res_net_units = int(np.ceil(config.n_mlp_layers / 2))
            estimated_z = multilayer_resnet(estimated_z, is_training, parameters, n_res_net_units,
                                            config.n_mlp_width, config.use_batch_norm, 'decoder_rn')
            sampled_z = multilayer_resnet(sampled_z, False, parameters, n_res_net_units,
                                          config.n_mlp_width, config.use_batch_norm, 'decoder_rn', True)
        else:
            estimated_z = multilayer_fcn(estimated_z, is_training, parameters, config.n_mlp_layers,
                                         config.n_mlp_width, config.use_batch_norm, 'decoder_fcn')
            sampled_z = multilayer_fcn(sampled_z, False, parameters, config.n_mlp_layers,
                                       config.n_mlp_width, config.use_batch_norm, 'decoder_fcn', True)

        estimated_z = tf.concat([estimated_z, parameters], 1) if parameters is not None else estimated_z
        sampled_z = tf.concat([sampled_z, parameters], 1) if parameters is not None else sampled_z
        y, layer = dense(estimated_z, n_outputs, activation_fn=None)
        y2, _ = dense(sampled_z, n_outputs, activation_fn=None, dense_layer=layer)
        return y, y2


def avb_discriminator(z, is_training, parameters, config):

    with tf.name_scope('discriminator'):
        d = multilayer_fcn(z, is_training, parameters, config.n_mlp_layers,
                           config.n_mlp_width, config.use_batch_norm, 'discriminator_fcn')

        d, _ = dense(d, 1, activation_fn=tf.nn.sigmoid)
        return d


def standardVAE(trainer, config, output_dir):
    with tf.name_scope('VAE'):
        n_outputs = trainer.out_pos_p.shape[1]
        rel_out_pos = (trainer.out_pos_p - trainer.out_pos_mean_p) * trainer.out_pos_stdinv_p
        z_mean, z_log_sigma2 = encoder(rel_out_pos, trainer.phase_p, None, config)
        vae_out, vae_out_gen = decoder(n_outputs, z_mean, z_log_sigma2, trainer.phase_p, trainer.latent_z, None, config)
        out_pos = vae_out[:, :trainer.out_pos_p.shape[1]]
        out_pos_gen = vae_out_gen[:, :trainer.out_pos_p.shape[1]]
        out_pos = out_pos / trainer.out_pos_stdinv_p + trainer.out_pos_mean_p
        out_pos_gen = out_pos_gen / trainer.out_pos_stdinv_p + trainer.out_pos_mean_p

    return {'out_pos': out_pos, 'out_pos_gen': out_pos_gen, 'z_mean': z_mean, 'z_log_sigma2': z_log_sigma2}
