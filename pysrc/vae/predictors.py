
import tensorflow as tf
import numpy as np

import vae.trainer
import vae.tf_utils

from vae.utils import shape_feat_name_to_num_coeff

import vae.global_config


def get_placeholder(dtype, shape=None, name=None):
    tensor_name = name + ':0'
    try:
        return tf.get_default_graph().get_tensor_by_name(tensor_name)
    except KeyError:
        return tf.placeholder(dtype, shape, name)


class PlaceholderManager:

    def __init__(self, dim=3):
        self.dim = dim
        self.in_pos_p = get_placeholder(tf.float32, [None, dim], 'inPos')
        self.in_normal_p = get_placeholder(tf.float32, [None, dim], 'inNormal')
        self.in_dir_p = get_placeholder(tf.float32, [None, dim], 'inDir')
        self.out_dir_p = get_placeholder(tf.float32, [None, dim], 'outDir')
        self.out_normal_p = get_placeholder(tf.float32, [None, dim], 'outNormal')
        self.out_pos_p = get_placeholder(tf.float32, [None, dim], 'outPos')
        self.out_pos_mean_p = get_placeholder(tf.float32, [dim], 'outPosMean')
        self.out_pos_stdinv_p = get_placeholder(tf.float32, [dim], 'outPosStdInv')
        self.absorption_prob_p = get_placeholder(tf.float32, [None, 1], 'absorptionProb')

        self.azimuth_transf_p = get_placeholder(tf.float32, [None, dim, dim], 'azimuthTransform')

        # material parameters
        self.albedo_p = get_placeholder(tf.float32, [None, 3], 'albedo')
        self.albedo_mean_p = get_placeholder(tf.float32, [3], 'albedoMean')
        self.albedo_stdinv_p = get_placeholder(tf.float32, [3], 'albedoStdInv')
        self.eff_albedo_p = get_placeholder(tf.float32, [None, 1], 'effAlbedo')
        self.eff_albedo_mean_p = get_placeholder(tf.float32, [1], 'effAlbedoMean')
        self.eff_albedo_stdinv_p = get_placeholder(tf.float32, [1], 'effAlbedoStdInv')
        self.g_p = get_placeholder(tf.float32, [None, 1], 'g')
        self.g_mean_p = get_placeholder(tf.float32, [1], 'gMean')
        self.g_stdinv_p = get_placeholder(tf.float32, [1], 'gStdInv')
        self.sigma_t_p = get_placeholder(tf.float32, [None, 3], 'sigmaT')
        self.sigma_t_mean_p = get_placeholder(tf.float32, [3], 'sigmaTMean')
        self.sigma_t_stdinv_p = get_placeholder(tf.float32, [3], 'sigmaTStdInv')
        self.ior_p = get_placeholder(tf.float32, [None, 1], 'ior')

        self.phase_p = get_placeholder(tf.bool, name='phase')
        self.dropout_keep_prob_p = get_placeholder(tf.float32, name='dropoutKeepProb')

        self.shape_features_p, self.shape_features_mean_p, self.shape_features_stdinv_p = None, None, None

    def create_shape_placeholders(self, shape_features_name):
        if shape_features_name:
            n_shape_coeffs = shape_feat_name_to_num_coeff(shape_features_name, self.dim)
            self.shape_features_p = get_placeholder(tf.float32, [None, n_shape_coeffs], 'shapeFeatures')
            self.shape_features_mean_p = get_placeholder(tf.float32, [n_shape_coeffs], 'shapeFeaturesMean')
            self.shape_features_stdinv_p = get_placeholder(tf.float32, [n_shape_coeffs], 'shapeFeaturesStdInv')
            self.poly_scale_factor_p = get_placeholder(tf.float32, [None, 1], 'polyScaleFactor')

    def create_placeholder(self, var_name, dtype, shape=None, name=None):
        self.__dict__[var_name] = get_placeholder(dtype, shape, name)


def vae_loss(x_in, x_rec,
             z_mean, z_log_sigma2, wae_z_enc,
             global_step, gen_loss_weight,
             gen_loss='l2',
             loss_clamp_val=0.0,
             use_wae_mmd=False,
             wae_random_enc=False,
             latent_loss_annealing=None,
             ph_manager=None,
             config=None):

    with tf.name_scope('loss'):
        if gen_loss == 'l2':
            generation_loss = tf.reduce_mean(tf.square(x_in - x_rec), axis=1)
            gen_loss_weight = gen_loss_weight / 100
        elif gen_loss == 'l1':
            generation_loss = tf.reduce_mean(tf.abs(x_in - x_rec), axis=1)
            gen_loss_weight = gen_loss_weight / 10
        elif gen_loss == 'huber':
            generation_loss = tf.reduce_mean(tf.losses.huber_loss(x_in, x_rec, delta=25.0))
            gen_loss_weight = gen_loss_weight / 100
        else:
            raise ValueError('Undefined loss {}'.format(gen_loss))

        if loss_clamp_val > 0:
            generation_loss = tf.minimum(generation_loss, loss_clamp_val)

        if config.scale_loss_by_kernel_epsilon:
            generation_loss = generation_loss * ph_manager.poly_scale_factor_p

        generation_loss = gen_loss_weight * generation_loss
        loss = generation_loss

        if use_wae_mmd:
            latent_loss = mmd_loss(wae_z_enc, tf.random_normal(tf.shape(wae_z_enc), 0, 1))
            if wae_random_enc:
                loss = loss + tf.reduce_mean(tf.abs(z_log_sigma2))
        else:
            latent_loss = 0.5 * tf.reduce_sum(tf.square(z_mean) +
                                              tf.exp(z_log_sigma2) - z_log_sigma2 - 1, axis=1)

        if latent_loss_annealing == 'linear':
            latent_loss_weight = tf.clip_by_value(global_step / 1000000, 0, 1)

        elif latent_loss_annealing == 'square':
            latent_loss_weight = tf.square(tf.clip_by_value(global_step / 1000000, 0, 1))

        elif latent_loss_annealing is None:
            latent_loss_weight = 1.0

        latent_loss = latent_loss * tf.cast(latent_loss_weight, tf.float32)
        loss = tf.reduce_mean(loss + latent_loss)
        d = {'loss': loss, 'generation_loss': generation_loss, 'latent_loss': latent_loss}

        if config.off_surface_penalty_weight > 0.0:
            poly_value = vae.tf_utils.eval_poly(x_rec, ph_manager.in_pos_p, -ph_manager.in_dir_p,
                                                ph_manager.shape_features_p, config.poly_order(),
                                                config.prediction_space == 'LS', ph_manager.poly_scale_factor_p)
            if config.off_surface_penalty_clamp > 0.0:
                off_surface_loss = config.off_surface_penalty_weight * tf.reduce_mean(tf.minimum(tf.abs(poly_value), config.off_surface_penalty_clamp))
            else:
                off_surface_loss = config.off_surface_penalty_weight * tf.reduce_mean(tf.abs(poly_value))

            d['loss'] += off_surface_loss
            d['off_surface_loss'] = off_surface_loss
        return d


def nice_loss(rec_u, log_jacobian):
    # Standard Gaussian Prior
    with tf.name_scope('loss'):
        prior_loss = tf.reduce_mean(0.5 * tf.square(rec_u) + np.log(2 * np.pi))
        data_loss = tf.reduce_mean(log_jacobian)
        loss = prior_loss + data_loss
    return {'loss': loss, 'log_jacobian': log_jacobian, 'prior_loss': prior_loss}


class ScatterPredictor:
    def __init__(self, ph_manager, config, output_dir):
        self.config = config
        self.prefix = 'scatter'
        self.ph_manager = ph_manager

        if config.use_point_net:
            ph_manager.create_placeholder('points_p', tf.float32, [None, config.n_point_net_points, 3], 'points')
            ph_manager.create_placeholder('points_mean_p', tf.float32, [3], 'pointsMean')
            ph_manager.create_placeholder('points_stdinv_p', tf.float32, [3], 'pointsStdInv')
            ph_manager.create_placeholder('point_normals_p', tf.float32, [None, config.n_point_net_points, 3], 'normals')
            ph_manager.create_placeholder('point_normals_mean_p', tf.float32, [3], 'normalsMean')
            ph_manager.create_placeholder('point_normals_stdinv_p', tf.float32, [3], 'normalsStdInv')
            ph_manager.create_placeholder('point_weights_p', tf.float32, [None, config.n_point_net_points], 'weights')
            ph_manager.create_placeholder('point_weights_mean_p', tf.float32, [1], 'weightsMean')
            ph_manager.create_placeholder('point_weights_stdinv_p', tf.float32, [1], 'weightsStdInv')
            ph_manager.create_placeholder('poly_scale_factor_p', tf.float32, [None, 1], 'polyScaleFactor')
        else:
            ph_manager.create_shape_placeholders(config.shape_features_name)
        ph_manager.create_placeholder('latent_z', tf.float32, [None, config.n_latent], 'scatterLatent')
        with tf.variable_scope(self.prefix):
            self.model_outputs = config.model(ph_manager, config, output_dir)

    def get_losses(self, global_step):
        if self.config.use_nice:
            losses = nice_loss(self.model_outputs['rec_u'], self.model_outputs['log_jacobian'])
            with tf.name_scope('loss/'):
                tf.summary.scalar('prior_loss', tf.reduce_mean(losses['prior_loss']))
                tf.summary.scalar('log_jacobian', tf.reduce_mean(losses['log_jacobian']))
                tf.summary.scalar('loss', losses['loss'])
        else:
            if self.config.use_epsilon_space:
                losses = vae_loss(self.model_outputs['ref_pos_eps'], self.model_outputs['out_pos_eps'], self.model_outputs['z_mean'],
                                self.model_outputs['z_log_sigma2'], self.model_outputs['z_encoded'], global_step, self.config.gen_loss_weight,
                                self.config.gen_loss, self.config.loss_clamp_val, self.config.use_wae_mmd,
                                self.config.wae_random_enc, self.config.latent_loss_annealing, self.ph_manager, self.config)
            else:
                losses = vae_loss(self.ph_manager.out_pos_p, self.model_outputs['out_pos'], self.model_outputs['z_mean'],
                                self.model_outputs['z_log_sigma2'], self.model_outputs['z_encoded'], global_step, self.config.gen_loss_weight,
                                self.config.gen_loss, self.config.loss_clamp_val, self.config.use_wae_mmd,
                                self.config.wae_random_enc, self.config.latent_loss_annealing, self.ph_manager, self.config)
            with tf.name_scope('loss/'):
                if self.config.off_surface_penalty_weight > 0.0:
                    tf.summary.scalar('off_surface_loss', tf.reduce_mean(losses['off_surface_loss']))

                tf.summary.scalar('generation_loss', tf.reduce_mean(losses['generation_loss']))
                tf.summary.scalar('latent_loss', tf.reduce_mean(losses['latent_loss']))
                tf.summary.scalar('loss', losses['loss'])
        return losses

    def dump_graph_info(self, filename):
        vae.tf_utils.dump_pbtxt_config(['scatter/out_pos_gen'], self.model_outputs['out_pos_gen'], self.config,
                                       filename + '_out_pos_gen_batched', 8, 'OutPosGenBatched')
        vae.tf_utils.dump_pbtxt_config(['scatter/out_pos_gen'], self.model_outputs['out_pos_gen'],
                                       self.config, filename + '_out_pos_gen', 1, 'OutPosGen')


class AbsorptionPredictor:
    def __init__(self, ph_manager, config, output_dir):
        self.config = config
        self.prefix = 'absorption'
        self.ph_manager = ph_manager

        if config.use_point_net:
            ph_manager.create_placeholder('points_p', tf.float32, [None, config.n_point_net_points, 3], 'points')
            ph_manager.create_placeholder('points_mean_p', tf.float32, [3], 'pointsMean')
            ph_manager.create_placeholder('points_stdinv_p', tf.float32, [3], 'pointsStdInv')
            ph_manager.create_placeholder('point_normals_p', tf.float32, [None, config.n_point_net_points, 3], 'normals')
            ph_manager.create_placeholder('point_normals_mean_p', tf.float32, [3], 'normalsMean')
            ph_manager.create_placeholder('point_normals_stdinv_p', tf.float32, [3], 'normalsStdInv')
            ph_manager.create_placeholder('point_weights_p', tf.float32, [None, config.n_point_net_points], 'weights')
            ph_manager.create_placeholder('point_weights_mean_p', tf.float32, [1], 'weightsMean')
            ph_manager.create_placeholder('point_weights_stdinv_p', tf.float32, [1], 'weightsStdInv')
            ph_manager.create_placeholder('poly_scale_factor_p', tf.float32, [None, 1], 'polyScaleFactor')
        else:
            ph_manager.create_shape_placeholders(config.shape_features_name)

        with tf.variable_scope(self.prefix):
            self.model_outputs = config.model(ph_manager, config, output_dir)

    def get_losses(self, global_step):
        absorption = self.model_outputs['absorption']
        with tf.name_scope('loss/'):
            if self.config.abs_loss == 'l2':
                absorption_prob_loss = tf.reduce_mean(tf.square(absorption - self.ph_manager.absorption_prob_p))
            elif self.config.abs_loss == 'l1':
                absorption_prob_loss = tf.reduce_mean(tf.abs(absorption - self.ph_manager.absorption_prob_p))
            elif self.config.abs_loss == 'crossentropy':
                absorption_prob_loss = -self.ph_manager.absorption_prob_p * \
                    tf.log(absorption) - (1 - self.ph_manager.absorption_prob_p) * tf.log(1 - absorption)
            elif self.config.abs_loss == 'classification':
                ref_label = tf.floor(self.config.n_abs_buckets * self.ph_manager.absorption_prob_p)
                absorption_prob_loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=ref_label, logits=self.model_outputs['logits'])

            tf.summary.scalar('loss', tf.reduce_mean(absorption_prob_loss))
        return {'loss': absorption_prob_loss}

    def dump_graph_info(self, filename):
        vae.tf_utils.dump_pbtxt_config(['absorption/absorption'], self.model_outputs['absorption'],
                                       self.config, filename + '_absorption', 1, 'Absorption')


class AngularScatterPredictor:
    def __init__(self, ph_manager, config, output_dir):
        self.config = config
        self.prefix = 'angular'
        self.ph_manager = ph_manager

        ph_manager.create_shape_placeholders(config.shape_features_name)
        ph_manager.create_placeholder('angular_latent_z', tf.float32, [None, config.n_latent], 'angularLatent')

        with tf.variable_scope(self.prefix):
            self.model_outputs = config.model(ph_manager, config, output_dir)

    def get_losses(self, global_step):
        if self.config.use_vmf:
            losses = {'loss': tf.reduce_mean(-self.model_outputs['log_pdf'])}
            with tf.name_scope('loss/'):
                tf.summary.scalar('loss', losses['loss'])
        elif self.config.use_nice:
            losses = nice_loss(self.model_outputs['rec_u'], self.model_outputs['log_jacobian'])
            with tf.name_scope('loss/'):
                tf.summary.scalar('prior_loss', tf.reduce_mean(losses['prior_loss']))
                tf.summary.scalar('log_jacobian', tf.reduce_mean(losses['log_jacobian']))
                tf.summary.scalar('loss', losses['loss'])
        else:
            losses = vae_loss(self.ph_manager.out_pos_p, self.model_outputs['out_pos'], self.model_outputs['z_mean'],
                              self.model_outputs['z_log_sigma2'], self.model_outputs['z_encoded'], global_step, self.config.gen_loss_weight,
                              self.config.gen_loss, self.config.loss_clamp_val, self.config.use_wae_mmd,
                              self.config.wae_random_enc, self.config.latent_loss_annealing, self.ph_manager, self.config)
            with tf.name_scope('loss/'):
                tf.summary.scalar('generation_loss', tf.reduce_mean(losses['generation_loss']))
                tf.summary.scalar('latent_loss', tf.reduce_mean(losses['latent_loss']))
                tf.summary.scalar('loss', losses['loss'])
        return losses

    def dump_graph_info(self, filename):
        vae.tf_utils.dump_pbtxt_config(['angular/out_dir_gen'], self.model_outputs['out_dir_gen'], self.config,
                                       filename + '_out_dir_gen_batched', 8, 'OutDirGenBatched')
        vae.tf_utils.dump_pbtxt_config(['angular/out_dir_gen'], self.model_outputs['out_dir_gen'],
                                       self.config, filename + '_out_dir_gen', 1, 'OutDirGen')
