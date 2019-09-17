import os

import numpy as np
import tensorflow as tf
from matplotlib.patches import Polygon
import time

import vae.utils
from utils.tensorflow import *
from vae.utils import *
from utils.mesh import sample_mesh
from vae.tf_utils import world_to_local, local_to_world, world_to_local_new, local_to_world_new
import vae.tf_utils

from mitsuba.core import *
import mitsuba.render
from vae.global_config import *

from models.nn import *
import models.nice
import models.vae

import utils.printing
import utils.mtswrapper

try:
    import matplotlib.pyplot as plt
except Exception as e:
    # See: https://stackoverflow.com/questions/4931376/generating-matplotlib-graphs-without-a-running-x-server/4935945#4935945
    import matplotlib
    matplotlib.use('Agg', force=True, warn=False)
    import matplotlib.pyplot as plt


def share_variables(func):
    """Decorator to easily share variables globally across scopes etc"""
    return tf.make_template(func.__name__, func, create_scope_now_=True)


def share_variables2(func, name):
    """Decorator to easily share variables globally across scopes etc"""
    return tf.make_template(name, func, create_scope_now_=True)


# @share_variables
def shared_preproc_mlp_impl(features, is_training):
    return models.nn.multilayer_fcn(features, is_training, None, 3, 32, False, 'shapemlp')

# @share_variables


def shared_preproc_mlp_2_impl(features, is_training):
    return models.nn.multilayer_fcn(features, is_training, None, 3, 64, False, 'shapemlp')


@share_variables
def baseline_point_net_shared(placeholders, config):
    return baseline_point_net(placeholders, config)


def recreate_shared_functions():
    globals()['shared_preproc_mlp'] = share_variables2(shared_preproc_mlp_impl, 'shared_preproc_mlp')
    globals()['shared_preproc_mlp_2'] = share_variables2(shared_preproc_mlp_2_impl, 'shared_preproc_mlp_2')


recreate_shared_functions()


def baseline_point_net(placeholders, config):
    reference_frame = -placeholders.in_dir_p if config.prediction_space == 'LS' else placeholders.in_normal_p
    in_pos = tf.expand_dims(placeholders.in_pos_p, 1)
    reference_frame = tf.expand_dims(reference_frame, 1)
    pts = world_to_local(in_pos, reference_frame,
                         placeholders.points_p, config.predict_in_tangent_space)
    pts = (pts - placeholders.points_mean_p) * placeholders.points_stdinv_p

    if config.scale_point_by_poly_scale:
        pts *= tf.expand_dims(placeholders.poly_scale_factor_p, -1)

    # Apply 1D convolutions to pts
    if config.point_net_use_normals:
        nors = world_to_local(tf.zeros_like(in_pos), reference_frame,
                              placeholders.point_normals_p, config.predict_in_tangent_space)

        cos_theta = nors[:, :, 2]
        nors = (nors - placeholders.point_normals_mean_p) * placeholders.point_normals_stdinv_p
        x = tf.concat([pts, nors], axis=-1)
    else:
        x = pts

    x = tf.expand_dims(x, 1)

    for i, s in enumerate(config.point_net_feature_sizes):
        # x = conv2d_layer(x, prev_s, s, 1)
        x = conv2d_layer_new(x, s, 1, f'conv{i}')
        # prev_s = s

    if config.point_net_normal_histogram:
        x = tf.reshape(x, [-1, config.n_point_net_points, config.point_net_feature_sizes[-1]])
        if config.point_net_use_weights:
            x, counts = vae.tf_utils.binned_average(
                x, cos_theta, [-1.01, 1.01], config.point_net_n_normal_bins, weights=placeholders.point_weights_p)
        else:
            x, counts = vae.tf_utils.binned_average(
                x, cos_theta, [-1.01, 1.01], config.point_net_n_normal_bins)

        return tf.reshape(x, [-1, config.point_net_n_normal_bins * x.shape[-1]])
    elif config.point_net_use_weights:
        w = placeholders.point_weights_p / \
            tf.maximum(tf.reduce_sum(placeholders.point_weights_p, 1, keepdims=True), 0.01)
        w = tf.expand_dims(w, 1)
        w = tf.expand_dims(w, -1)
        return tf.reduce_sum(x * w, (1, 2))
    else:
        return tf.reduce_mean(x, (1, 2))


def polyCnn(x, placeholders, config):

    x = utils.tensorflow.conv3d_layer(x, 1, 4, 3)
    x = utils.tensorflow.max_pool_2x2x2(x)
    x = utils.tensorflow.conv3d_layer(x, 4, 8, 3)
    x = utils.tensorflow.max_pool_2x2x2(x)
    return x


def preprocess_features(placeholders, config):
    # 1. Transform all the features using fixed transforms

    if config.use_similarity_theory:
        albedo_p = vae.utils.get_alphap(placeholders.albedo_p, placeholders.g_p, placeholders.sigma_t_p)
        albedo_norm = (vae.tf_utils.albedo_to_effective_albedo(albedo_p) -
                       placeholders.eff_albedo_mean_p) * placeholders.eff_albedo_stdinv_p
    elif config.use_eff_albedo:
        albedo_norm = (placeholders.eff_albedo_p - placeholders.eff_albedo_mean_p) * placeholders.eff_albedo_stdinv_p
    else:
        albedo_norm = (placeholders.albedo_p - placeholders.albedo_mean_p) * placeholders.albedo_stdinv_p

    g_norm = (placeholders.g_p - placeholders.g_mean_p) * placeholders.g_stdinv_p
    ior_norm = 2.0 * (placeholders.ior_p - 1.25)

    if config.shape_features_name:
        shape_features = (placeholders.shape_features_p - placeholders.shape_features_mean_p) * \
            placeholders.shape_features_stdinv_p

        # Regularization: Add some noise to the polynomial coefficients
        if config.add_noise_to_poly_coeffs:
            def add_noise(x):
                return x + tf.truncated_normal(tf.shape(x), 0.0, config.feat_noise_variance)

            def do_nothing(x):
                return x
            shape_features = tf.cond(placeholders.phase_p, lambda: add_noise(
                shape_features), lambda: do_nothing(shape_features))

        # 2. Optional: Apply a scaled sigmoid transform to the polynomial features to suppress outliers
        if config.sigmoid_features:
            shape_features = tf.tanh(config.sigmoid_scale_factor * shape_features) / config.sigmoid_scale_factor

        if config.eval_poly_feature:
            r = config.poly_eval_range
            x, y, z = np.meshgrid(np.linspace(-r, r, config.poly_eval_steps),
                                  np.linspace(-r, r, config.poly_eval_steps),
                                  np.linspace(-r, r, config.poly_eval_steps), indexing='ij')
            points = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1).astype(np.float32)[None, :, :]
            points = tf.constant(points)
            points = tf.tile(points, [tf.shape(placeholders.shape_features_p)[0], 1, 1])
            shape_features = vae.tf_utils.eval_poly(points, None, None, placeholders.shape_features_p,
                                                    config.poly_order(), tangent_space=False, scale=1.0)[:, :, 0]

            if config.sigmoid_features:
                shape_features = tf.tanh(config.sigmoid_scale_factor * shape_features) / config.sigmoid_scale_factor

            if config.binary_features:
                shape_features = tf.cast(shape_features > 0, tf.float32)

            if config.poly_conv_net:
                shape_features = tf.reshape(shape_features, [tf.shape(shape_features)[0], config.poly_eval_steps,
                                                             config.poly_eval_steps, config.poly_eval_steps, 1])
                shape_features = config.poly_conv_net(shape_features, placeholders, config)
                n_out_elems = shape_features.shape[1] * shape_features.shape[2] * \
                    shape_features.shape[3] * shape_features.shape[4]
                shape_features = tf.reshape(shape_features, [tf.shape(shape_features)[0], n_out_elems])

    if config.use_point_net:
        shape_features = config.point_net(placeholders, config)

    # 3. Optional: Get in_dir coordinats in tangent space
    in_dir_ts = world_to_local(tf.zeros_like(placeholders.in_pos_p),
                               placeholders.in_normal_p, -placeholders.in_dir_p, True)
    in_dir_proj = in_dir_ts[:, :2]
    in_dir_angle = tf.acos(in_dir_ts[:, 2]) - np.pi / 4

    if config.pass_in_dir and not config.rotate_features:
        features = tf.concat([shape_features, tf.expand_dims(albedo_norm[:, 0], 1), g_norm,
                              ior_norm, in_dir_proj, tf.expand_dims(in_dir_angle, 1)], 1)
    else:
        features = tf.concat([shape_features, tf.expand_dims(albedo_norm[:, 0], 1), g_norm, ior_norm], 1)

    # 4. Optionally: Run a preprocessing block over the features
    if config.shape_feat_net:
        features = config.shape_feat_net(features, placeholders.phase_p)

    if config.pass_in_dir_after_preprocess:
        features = tf.concat([features, in_dir_proj, tf.expand_dims(in_dir_angle, 1)], 1)

    batch_size = tf.shape(features)[0]
    if config.rotate_features:
        # Assuming that features as dim [?, 3 * k]
        features_reshaped = tf.reshape(features, [batch_size, features.shape[1] // 3, 3])
        # Rotate these points
        features_ls = world_to_local(tf.expand_dims(tf.zeros_like(placeholders.in_pos_p), 1),
                                     tf.expand_dims(in_dir_ts, 1), features_reshaped, True)
        features = tf.reshape(features_ls, [batch_size, features.shape[1]])
        if config.pass_in_dir:
            features = tf.concat([features, in_dir_proj, tf.expand_dims(in_dir_angle, 1)], 1)
    return features


def absorptionMlp(x, trainer, config):
    return multilayer_fcn(x, trainer.phase_p, None,
                          config.n_abs_layers, config.n_abs_width, False, 'mlp', regularizer=config.abs_regularizer)


def absorptionResNet(x, trainer, config):
    n_res_net_units = int(np.ceil(config.n_abs_layers / 2))
    return multilayer_resnet(x, trainer.phase_p, None,
                             n_res_net_units, config.n_abs_width, True, 'mlp', regularizer=config.abs_regularizer)


def absorptionMlpDropout(x, trainer, config):
    return multilayer_fcn(x, trainer.phase_p, None,
                          config.n_abs_layers, config.n_abs_width, False, 'mlp',
                          use_dropout=True, dropout_keep_prob=trainer.dropout_keep_prob_p, regularizer=config.abs_regularizer)


def absorptionPredictor(trainer, config, output_dir):
    """MLP which predicts the absorption coefficient based on albedo and shape descriptor"""
    model_outputs = {}
    features = preprocess_features(trainer, config)
    model_outputs['features'] = features
    if config.abs_use_res_net:
        n_res_net_units = int(np.ceil(config.n_abs_layers / 2))
        predicted = multilayer_resnet(features, trainer.phase_p, None, n_res_net_units, config.n_abs_width,
                                      True, 'mlp', regularizer=config.abs_regularizer)
    else:
        predicted = config.absorption_model(features, trainer, config)

    if config.abs_bounce_distribution == 'gamma':
        gamma, _ = dense(predicted, 2, activation_fn=None, out_name='absorption')
        gamma = tf.exp(gamma)
        predicted = 1.0 - tf.pow(1.0 / (1.0 - gamma[:, 0] * tf.log(trainer.albedo_p[:, 0])), gamma[:, 1])
    elif config.abs_loss == 'classification':
        logits, _ = dense(predicted, config.n_abs_buckets, activation_fn=None, out_name='absorption')
        model_outputs['logits'] = logits
        label = tf.nn.softmax(logits)
        label = tf.reduce_sum(label * (np.arange(config.n_abs_buckets, dtype=np.float32) + 0.5), -1)
        predicted = label / config.n_abs_buckets
    else:
        predicted, _ = dense(predicted, 1, activation_fn=tf.sigmoid, out_name='absorption')

    if config.abs_predict_diff:
        predicted = trainer.eff_albedo_p + predicted - tf.constant(0.5)
    model_outputs['absorption'] = predicted
    return model_outputs


def baselineCNN(trainer, config, output_dir):
    # Apply some convolutional layers to the local_grid
    discretized_geometry = vae.tf_utils.extract_voxel_grid(trainer, output_dir)
    filtered_geometry = config.convnet(discretized_geometry)
    features = tf.reshape(filtered_geometry,
                          [-1, filtered_geometry.shape[1] * filtered_geometry.shape[2] * filtered_geometry.shape[3]])

    reference_frame = -trainer.in_dir_p if config.prediction_space == 'LS' else trainer.in_normal_p
    rel_out_pos = world_to_local(trainer.in_pos_p, reference_frame,
                                 trainer.out_pos_p, config.predict_in_tangent_space)

    rel_out_pos = (rel_out_pos - trainer.out_pos_mean_p) * trainer.out_pos_stdinv_p

    if config.use_wae_mmd:
        z_mean = None
        z_log_sigma2 = None
        z_encoded = models.vae.encoder(rel_out_pos, trainer.phase_p, features, config)
        vae_out_pos, vae_out_pos_gen = models.vae.decoder(
            trainer.out_pos_p.shape[1], None, None, trainer.phase_p, trainer.latent_z, features, config, z_encoded)
    else:
        z_mean, z_log_sigma2 = models.vae.encoder(rel_out_pos, trainer.phase_p, features, config)
        vae_out_pos, vae_out_pos_gen = models.vae.decoder(
            trainer.out_pos_p.shape[1], z_mean, z_log_sigma2, trainer.phase_p, trainer.latent_z, features, config)
        z_encoded = None

    vae_out_pos = vae_out_pos / trainer.out_pos_stdinv_p + trainer.out_pos_mean_p
    vae_out_pos_gen = vae_out_pos_gen / trainer.out_pos_stdinv_p + trainer.out_pos_mean_p
    vae_out_pos = local_to_world(trainer.in_pos_p, reference_frame,
                                 vae_out_pos, config.predict_in_tangent_space)
    vae_out_pos_gen = local_to_world(trainer.in_pos_p, reference_frame,
                                     vae_out_pos_gen, config.predict_in_tangent_space)

    return {'out_pos': vae_out_pos, 'out_pos_gen': vae_out_pos_gen, 'z_encoded': z_encoded,
            'z_mean': z_mean, 'z_log_sigma2': z_log_sigma2}


def baselineNICE(trainer, config, output_dir):

    model_outputs = {}
    features = preprocess_features(trainer, config)
    model_outputs['features'] = features
    reference_frame = -trainer.in_dir_p if config.prediction_space == 'LS' else trainer.in_normal_p
    rel_out_pos = world_to_local(trainer.in_pos_p, reference_frame,
                                 trainer.out_pos_p, config.predict_in_tangent_space)
    rel_out_pos = (rel_out_pos - trainer.out_pos_mean_p) * trainer.out_pos_stdinv_p

    if config.use_res_net:
        net_type = 'resnet'
    elif config.use_legacy_res_net:
        net_type = 'legacyresnet'
    else:
        net_type = 'mlp'

    # Compute PDF of sample to do maximum-likelihood
    out_pos_gen, rec_u, log_jacobian = models.nice.niceModel(
        rel_out_pos, trainer.latent_z, False, config.n_coupling_layers, config.n_mlp_layers,
        config.n_mlp_width, trainer.phase_p, features, config.nice_first_layer_feats,
        net_type, use_coord_offset=config.use_coupling_coord_offset, dim=3)
    out_pos_gen = out_pos_gen / trainer.out_pos_stdinv_p + trainer.out_pos_mean_p
    out_pos_gen = local_to_world(trainer.in_pos_p, reference_frame,
                                 out_pos_gen, config.predict_in_tangent_space, 'out_pos_gen')

    model_outputs.update({'out_pos_gen': out_pos_gen, 'log_jacobian': log_jacobian, 'rec_u': rec_u})
    return model_outputs


def baselineShapeDescriptor(trainer, config, output_dir):
    model_outputs = {}
    features = preprocess_features(trainer, config)
    model_outputs['features'] = features

    if config.prediction_space == 'AS':
        rel_out_pos = trainer.out_pos_p - trainer.in_pos_p
        rel_out_pos = tf.matmul(trainer.azimuth_transf_p, tf.expand_dims(rel_out_pos, -1))[:, :, 0]
    else:
        reference_frame = -trainer.in_dir_p if config.prediction_space == 'LS' else trainer.in_normal_p
        rel_out_pos = world_to_local(trainer.in_pos_p, reference_frame,
                                     trainer.out_pos_p, config.predict_in_tangent_space)

    sigma_t_p = vae.utils.get_sigmatp(trainer.albedo_p, trainer.g_p, trainer.sigma_t_p)

    if config.use_epsilon_space:
        rel_out_pos *= trainer.poly_scale_factor_p
        ref_out_pos_eps = rel_out_pos
    elif config.use_similarity_theory and not config.scale_point_by_poly_scale:
        rel_out_pos *= sigma_t_p
    elif config.scale_point_by_poly_scale:
        rel_out_pos *= trainer.poly_scale_factor_p

    if config.use_outpos_statistics:
        rel_out_pos = (rel_out_pos - trainer.out_pos_mean_p) * trainer.out_pos_stdinv_p

    input_sample = rel_out_pos
    n_outputs = trainer.out_pos_p.shape[1]

    if config.use_wae_mmd:
        z_mean, z_log_sigma2 = None, None
        z_encoded = models.vae.encoder(input_sample, trainer.phase_p, features, config)
        vae_out, vae_out_gen = models.vae.decoder(
            n_outputs, None, None, trainer.phase_p, trainer.latent_z, features, config, z_encoded)
    else:
        z_mean, z_log_sigma2 = models.vae.encoder(input_sample, trainer.phase_p, features, config)
        vae_out, vae_out_gen = models.vae.decoder(
            n_outputs, z_mean, z_log_sigma2, trainer.phase_p, trainer.latent_z, features, config)
        z_encoded = None

    out_pos = vae_out[:, :trainer.out_pos_p.shape[1]]
    out_pos_gen = vae_out_gen[:, :trainer.out_pos_p.shape[1]]

    if config.use_outpos_statistics:
        out_pos = out_pos / trainer.out_pos_stdinv_p + trainer.out_pos_mean_p
        out_pos_gen = out_pos_gen / trainer.out_pos_stdinv_p + trainer.out_pos_mean_p

    out_pos_gen_unproj = tf.identity(out_pos_gen)

    # Approximate surface projection
    if config.surface_projection_method == 'gd':
        rate = 5.0
        for i in range(config.surface_projection_iterations):
            value = vae.tf_utils.eval_poly(out_pos, None, None, trainer.shape_features_p,
                                           config.poly_order(), tangent_space=False, scale=trainer.poly_scale_factor_p)
            grad = vae.tf_utils.tf_eval_poly_gradient(
                out_pos, None, None, trainer.shape_features_p, tangent_space=False,
                scale=trainer.poly_scale_factor_p, poly_order=config.poly_order())
            tf.check_numerics(grad, "grad is nan", name=None)
            out_pos = out_pos - tf.clip_by_value(rate * grad * tf.sign(value) * trainer.poly_scale_factor_p, -
                                                 config.surface_projection_clip_range, config.surface_projection_clip_range)

        for i in range(config.surface_projection_iterations):
            value = vae.tf_utils.eval_poly(out_pos_gen, None, None, trainer.shape_features_p,
                                           config.poly_order(), tangent_space=False, scale=trainer.poly_scale_factor_p)
            grad = vae.tf_utils.tf_eval_poly_gradient(
                out_pos_gen, None, None, trainer.shape_features_p, tangent_space=False, scale=trainer.poly_scale_factor_p, poly_order=config.poly_order())
            tf.check_numerics(grad, "grad is nan", name=None)
            out_pos_gen = out_pos_gen - tf.clip_by_value(rate * grad * tf.sign(
                value) * trainer.poly_scale_factor_p, -config.surface_projection_clip_range, config.surface_projection_clip_range)

    if config.use_epsilon_space:
        model_outputs['ref_pos_eps'] = ref_out_pos_eps
        model_outputs['out_pos_eps'] = out_pos
        out_pos = out_pos / trainer.poly_scale_factor_p
        out_pos_gen /= trainer.poly_scale_factor_p
        out_pos_gen_unproj /= trainer.poly_scale_factor_p
    elif config.use_similarity_theory and not config.scale_point_by_poly_scale:
        out_pos /= sigma_t_p
        out_pos_gen /= sigma_t_p
        out_pos_gen_unproj /= sigma_t_p
    elif config.scale_point_by_poly_scale:
        out_pos /= trainer.poly_scale_factor_p
        out_pos_gen /= trainer.poly_scale_factor_p
        out_pos_gen_unproj /= trainer.poly_scale_factor_p

    if config.prediction_space == 'AS':
        out_pos = tf.identity(tf.matmul(trainer.azimuth_transf_p, tf.expand_dims(out_pos, -1), transpose_a=True)[:, :, 0] + trainer.in_pos_p, 'out_pos')
        out_pos_gen = tf.identity(tf.matmul(trainer.azimuth_transf_p, tf.expand_dims(out_pos_gen, -1), transpose_a=True)[:, :, 0] + trainer.in_pos_p, 'out_pos_gen')
        out_pos_gen_unproj = out_pos_gen
    else:
        out_pos = local_to_world(trainer.in_pos_p, reference_frame,
                                 out_pos, config.predict_in_tangent_space, 'out_pos')
        out_pos_gen = local_to_world(trainer.in_pos_p, reference_frame,
                                     out_pos_gen, config.predict_in_tangent_space, 'out_pos_gen')
        out_pos_gen_unproj = local_to_world(trainer.in_pos_p, reference_frame,
                                            out_pos_gen_unproj, config.predict_in_tangent_space, 'out_pos_gen_unproj')

    model_outputs.update({'out_pos': out_pos, 'out_pos_gen': out_pos_gen,
                          'out_pos_gen_unproj': out_pos_gen_unproj,
                          'z_encoded': z_encoded, 'z_mean': z_mean, 'z_log_sigma2': z_log_sigma2})
    return model_outputs


def projectiveNet(trainer, config, output_dir):
    features = preprocess_features(trainer, config)

    reference_frame = -trainer.in_dir_p if config.prediction_space == 'LS' else trainer.in_normal_p
    rel_out_pos = world_to_local(trainer.in_pos_p, reference_frame,
                                 trainer.out_pos_p, config.predict_in_tangent_space)
    # rel_out_pos = (rel_out_pos - trainer.out_pos_mean_p) * trainer.out_pos_stdinv_p

    n_outputs = trainer.out_pos_p.shape[1]
    z_mean, z_log_sigma2 = models.vae.encoder(rel_out_pos, trainer.phase_p, features, config)

    samples = tf.random_normal([tf.shape(z_mean)[0], config.n_latent], 0, 1, dtype=tf.float32)
    estimated_z = z_mean + (tf.exp(z_log_sigma2 / 2) * samples)

    decoder = tf.make_template('projectiveDecoder', models.vae.projectiveDecoder)
    out_pos = decoder(estimated_z + trainer.out_pos_mean_p, n_outputs, trainer.phase_p, features, config, trainer)
    out_pos_gen = decoder(trainer.latent_z + trainer.out_pos_mean_p, n_outputs,
                          trainer.phase_p, features, config, trainer)

    # out_pos = out_pos / trainer.out_pos_stdinv_p + trainer.out_pos_mean_p
    # out_pos_gen = out_pos_gen / trainer.out_pos_stdinv_p + trainer.out_pos_mean_p
    out_pos = local_to_world(trainer.in_pos_p, reference_frame,
                             out_pos, config.predict_in_tangent_space, 'out_pos')
    out_pos_gen = local_to_world(trainer.in_pos_p, reference_frame,
                                 out_pos_gen, config.predict_in_tangent_space, 'out_pos_gen')
    return {'out_pos': out_pos, 'out_pos_gen': out_pos_gen,
            'out_pos_gen_unproj': out_pos_gen, 'z_encoded': None,
            'z_mean': z_mean, 'z_log_sigma2': z_log_sigma2}


def baselineAngularScatter(trainer, config, output_dir):
    features = preprocess_features(trainer, config)

    # Convert outgoing direction into the outgoing tangent space
    local_out_dir = world_to_local(tf.zeros_like(trainer.in_pos_p), trainer.out_normal_p, trainer.out_dir_p, True)

    # For now, just try to reconstruct the projection on x, y components. Later: Apply concentric mapping here
    # Multiply with ior ratio to implicitly account for critical angle
    input_sample = local_out_dir[:, :2] * trainer.ior_p

    # Compute PDF of sample to do maximum-likelihood
    sampled_x, rec_u, log_jacobian = models.nice.niceModel(
        input_sample, trainer.angular_latent_z, False, config.n_coupling_layers, config.n_mlp_layers,
        config.n_mlp_width, trainer.phase_p, features, config.nice_first_layer_feats, config.use_res_net)

    out_dir_gen = sampled_x / trainer.ior_p
    z = tf.sqrt(tf.clip_by_value(1 - tf.reduce_sum(out_dir_gen ** 2, axis=1), 0.0, 1.0))
    out_dir_gen = tf.stack([out_dir_gen[:, 0], out_dir_gen[:, 1], z], axis=1)
    out_dir_gen = local_to_world(tf.zeros_like(trainer.in_pos_p), trainer.out_normal_p,
                                 out_dir_gen, True, 'out_dir_gen')

    return {'out_dir_gen': out_dir_gen, 'log_jacobian': log_jacobian, 'rec_u': rec_u}


def vmfAngularScatter(predictor, config, output_dir):

    features = preprocess_features(predictor, config)

    # Relative position compared to input
    reference_frame = -predictor.in_dir_p if config.prediction_space == 'LS' else predictor.in_normal_p
    rel_out_pos = world_to_local(predictor.in_pos_p, reference_frame,
                                 predictor.out_pos_p, config.predict_in_tangent_space)
    rel_out_pos = (rel_out_pos - predictor.out_pos_mean_p) * predictor.out_pos_stdinv_p

    # Convert outgoing direction into light space
    local_out_dir = world_to_local(tf.zeros_like(predictor.in_pos_p), reference_frame,
                                   predictor.out_dir_p, config.predict_in_tangent_space)

    # Apply a fully connected network to predict parameters of vMF distribution: direction mu, concentration k
    nn_output = models.nn.multilayer_fcn(features, predictor.phase_p, None,
                                         config.n_mlp_layers, config.n_mlp_width, False, 'mlp')

    vmf_params = tf.contrib.layers.fully_connected(nn_output, 4, scope=f'fc_final', activation_fn=None)

    vmf_dir = tf.nn.tanh(vmf_params[:, :3])
    vmf_dir = vmf_dir / tf.sqrt(tf.reduce_sum(tf.square(vmf_dir), axis=1, keepdims=True))
    vmf_k = tf.nn.relu(vmf_params[:, 3]) + 0.01

    # pdf = vmf_k / (2 * np.pi * (1 - tf.exp(-2 * vmf_k))) * tf.exp(vmf_k * (tf.reduce_sum(vmf_dir * local_out_dir, axis=1, keepdims=True) - 1))
    log_pdf = tf.log(vmf_k / (2 * np.pi * (1 - tf.exp(-2 * vmf_k)))) + vmf_k * \
        (tf.reduce_sum(vmf_dir * local_out_dir, axis=1) - 1)

    # Sampling code
    u0 = predictor.angular_latent_z[:, 0]
    u1 = predictor.angular_latent_z[:, 1]

    u0 = 2 * np.pi * u0
    w = 1 + 1.0 / vmf_k * tf.log(u1 + (1 - u1) * tf.exp(-2 * vmf_k))
    sqrt_w = tf.sqrt(1 - w * w)
    out_dir_gen = tf.stack([sqrt_w * tf.cos(u0), sqrt_w * tf.sin(u0), w], axis=1)

    t1, t2 = vae.tf_utils.onb_duff(vmf_dir)

    out_dir_gen = tf.expand_dims(out_dir_gen[..., 0], -1) * t1 + \
        tf.expand_dims(out_dir_gen[..., 1], -1) * t2 + \
        tf.expand_dims(out_dir_gen[..., 2], -1) * vmf_dir

    # Convert to world space
    out_dir_gen = local_to_world(tf.zeros_like(predictor.in_pos_p), reference_frame,
                                 out_dir_gen, config.predict_in_tangent_space, 'out_dir_gen')

    vmf_dir_ws = local_to_world(tf.zeros_like(predictor.in_pos_p), reference_frame,
                                vmf_dir, config.predict_in_tangent_space, 'vmf_dir_ws')

    return {'out_dir_gen': out_dir_gen, 'log_pdf': log_pdf, 'vmf_k': vmf_k, 'vmf_dir': vmf_dir, 'vmf_dir_ws': vmf_dir_ws}


def debugAngularScatter(trainer, config, output_dir):
    batch_size = tf.shape(trainer.in_pos_p)[0]
    out_dir_gen = tf.zeros((batch_size, 2))
    z = tf.sqrt(tf.clip_by_value(1 - tf.reduce_sum(out_dir_gen ** 2, axis=1), 0.0, 1.0))
    out_dir_gen = tf.stack([out_dir_gen[:, 0], out_dir_gen[:, 1], z], axis=1)
    out_dir_gen = local_to_world(tf.zeros_like(trainer.in_pos_p), trainer.out_normal_p,
                                 out_dir_gen, True, 'out_dir_gen')

    rec_u = tf.zeros((batch_size, 2))
    log_jacobian = tf.Variable(tf.ones(4), 'debug_var')

    return {'out_dir_gen': out_dir_gen, 'log_jacobian': log_jacobian, 'rec_u': rec_u}


def generate_new_samples_feed_dict(predictor, in_pos, in_normal, in_direction, shape_features, albedo,
                                   sigma_t, g, eta, feature_statistics, config, out_pos=None, out_normals=None,
                                   points=None, point_normals=None, point_weights=None, adjusted_in_dir=None, poly_normal=None):
    feed_dict = {}
    feed_dict[predictor.in_pos_p] = in_pos

    feed_dict[predictor.in_normal_p] = in_normal
    feed_dict[predictor.in_dir_p] = in_direction

    if adjusted_in_dir is not None:
        feed_dict[predictor.azimuth_transf_p] = utils.transforms.to_azimuth_space(-adjusted_in_dir, poly_normal)
    else:
        feed_dict[predictor.azimuth_transf_p] = utils.transforms.to_azimuth_space(-in_direction, in_normal)

    feed_dict[predictor.phase_p] = False
    if config.shape_features_name:
        feed_dict[predictor.shape_features_p] = shape_features
        f = config.shape_features_name
        feed_dict[predictor.shape_features_mean_p] = feature_statistics['{}_mean'.format(f)]
        feed_dict[predictor.shape_features_stdinv_p] = feature_statistics['{}_stdinv'.format(f)]
        feed_dict[predictor.poly_scale_factor_p] = vae.utils.get_poly_scale_factor(
            vae.utils.kernel_epsilon(g, sigma_t, albedo))

    if config.use_point_net:
        feed_dict[predictor.points_p] = points
        feed_dict[predictor.points_mean_p] = feature_statistics['points{}_mean'.format(config.prediction_space)]
        feed_dict[predictor.points_stdinv_p] = feature_statistics['points{}_mean'.format(config.prediction_space)]
        feed_dict[predictor.point_normals_p] = point_normals
        feed_dict[predictor.point_normals_mean_p] = feature_statistics['pointNormals{}_mean'.format(
            config.prediction_space)]
        feed_dict[predictor.point_normals_stdinv_p] = feature_statistics['pointNormals{}_mean'.format(
            config.prediction_space)]
        feed_dict[predictor.point_weights_p] = point_weights
        feed_dict[predictor.poly_scale_factor_p] = vae.utils.get_poly_scale_factor(
            vae.utils.kernel_epsilon(g, sigma_t, albedo))

    if 'outPosRel{}_mean'.format(config.prediction_space) in feature_statistics:
        feed_dict[predictor.out_pos_mean_p] = feature_statistics['outPosRel{}_mean'.format(config.prediction_space)]
        feed_dict[predictor.out_pos_stdinv_p] = feature_statistics['outPosRel{}_stdinv'.format(config.prediction_space)]

    feed_dict[predictor.albedo_p] = albedo
    feed_dict[predictor.albedo_mean_p] = feature_statistics['albedo_mean']
    feed_dict[predictor.albedo_stdinv_p] = feature_statistics['albedo_stdinv']
    feed_dict[predictor.eff_albedo_p] = vae.utils.albedo_to_effective_albedo(albedo[..., 0][..., np.newaxis])
    feed_dict[predictor.eff_albedo_mean_p] = feature_statistics['effAlbedo_mean']
    feed_dict[predictor.eff_albedo_stdinv_p] = feature_statistics['effAlbedo_stdinv']
    feed_dict[predictor.g_p] = g
    feed_dict[predictor.g_mean_p] = feature_statistics['g_mean']
    feed_dict[predictor.g_stdinv_p] = feature_statistics['g_stdinv']

    feed_dict[predictor.ior_p] = eta
    feed_dict[predictor.sigma_t_p] = sigma_t
    feed_dict[predictor.sigma_t_mean_p] = feature_statistics['sigmaT_mean']
    feed_dict[predictor.sigma_t_stdinv_p] = feature_statistics['sigmaT_stdinv']

    if out_pos is not None:
        feed_dict[predictor.out_pos_p] = out_pos

    if out_normals is not None:
        feed_dict[predictor.out_normal_p] = out_normals

    return feed_dict


def extract_shape_features(config, in_pos, scene, constraint_kdtree, in_dir, g, sigma_t,
                           albedo, rotate_poly, mesh, in_normal,
                           use_legacy_kernel, kdtree_threshold, fit_regularization, use_hard_constraint):
    if use_legacy_kernel:
        print("Unsupported: Legacy kernel is deprecated")
    # Extract local geometry features
    if config.use_sh_coeffs:
        raise ValueError("Not Supported")
    elif config.shape_features_name:
        coeffs, coeffs_ws, pos_constraints, nor_constraints, adjusted_in_dir, poly_normal = extract_poly_shape_features(config, in_pos, constraint_kdtree, in_dir, g,
                                                                               sigma_t, albedo, rotate_poly, in_normal,
                                                                               kdtree_threshold, fit_regularization, use_hard_constraint)
        return {'features': 'poly', 'coeffs': coeffs, 'coeffs_ws': coeffs_ws, 'pos_constraints': pos_constraints, 'nor_constraints': nor_constraints,
                'adjusted_in_dir': adjusted_in_dir, 'poly_normal': poly_normal}
    elif config.use_point_net:
        points, point_normals, point_weights = extract_point_features(config, in_pos, constraint_kdtree, in_dir, g,
                                                                      sigma_t, albedo, rotate_poly, mesh, in_normal,
                                                                      use_legacy_kernel)

        return {'features': 'points', 'points': points, 'point_normals': point_normals, 'point_weights': point_weights}


def extract_poly_shape_features(config, in_pos, constraint_kdtree, in_dir, g, sigma_t,
                                albedo, rotate_poly, in_normal, kdtree_threshold, fit_regularization, use_hard_constraint=True):

    poly_order = extract_poly_order_from_feat_name(config.shape_features_name)
    options = {
        'regularization': fit_regularization,
        'hardSurfaceConstraint': use_hard_constraint,
        'order': poly_order,
        'kdtree_threshold': kdtree_threshold,
        'useLightspace': False
    }
    coeffs_ws, pos_constraints, nor_constraints = utils.mtswrapper.fitPolynomial(
        constraint_kdtree, in_pos, -in_dir, sigma_t, g, albedo, options, normal=in_normal)

    # Optional: adjust the incoming ray direction to the polynomial normal 
    scale_factor = vae.utils.get_poly_scale_factor(vae.utils.kernel_epsilon(g, sigma_t, albedo))
    adjusted_in_dir, poly_normal = utils.mtswrapper.adjust_ray_direction_for_polynomial(coeffs_ws, in_pos, in_dir, in_normal, scale_factor)

    if config.polynomial_space == 'LS':
        coeffs = utils.mtswrapper.rotate_polynomial(coeffs_ws, -in_dir, poly_order)
    elif config.polynomial_space == 'TS':
        coeffs = utils.mtswrapper.rotate_polynomial(coeffs_ws, in_normal, poly_order)
    elif config.polynomial_space == 'AS':
        in_normal = poly_normal 
        in_dir = adjusted_in_dir
        coeffs = utils.mtswrapper.rotate_polynomial_azimuth(coeffs_ws, -in_dir, in_normal, poly_order)
    else:
        coeffs = coeffs_ws
    if config.polynomial_space == 'AS' and not rotate_poly:
        raise ValueError("unsupported: use rotate poly with azimuth space")
    if config.polynomial_space == 'LS' and not rotate_poly:
        options['useLightspace'] = True
        coeffs, pos_constraints, nor_constraints = utils.mtswrapper.fitPolynomial(
            constraint_kdtree, in_pos, -in_dir, sigma_t, g, albedo, options, normal=in_normal)
    return coeffs, coeffs_ws, pos_constraints, nor_constraints, adjusted_in_dir, poly_normal


def extract_point_features(config, in_pos, constraint_kdtree, in_dir, g, sigma_t, albedo, rotate_poly, mesh, in_normal, use_legacy_kernel):
    in_pos = in_pos.ravel()
    in_pos_mts = Point3(float(in_pos[0]), float(in_pos[1]), float(in_pos[2]))
    effective_epsilon = vae.utils.kernel_epsilon(g, sigma_t, albedo)
    pts, nors, weights = mitsuba.render.Volpath3D.getLocalPoints(
        in_pos_mts, effective_epsilon, 'gaussian', constraint_kdtree)
    n_points = config.n_point_net_points
    pts = vae.utils.mts_to_np(pts)
    nors = vae.utils.mts_to_np(nors)
    weights = np.exp(-np.sum((in_pos - pts) ** 2, axis=1) / (2 * effective_epsilon))
    pt_indices = utils.math.weighted_sampling_without_replacement(weights.tolist(), n_points)
    pts = pts[pt_indices][None, :, :]
    nors = nors[pt_indices][None, :, :]
    weights = weights[pt_indices][None, :]
    if pts.shape[0] < n_points:  # Pad the arrays to n_points
        pts = np.concatenate([pts, np.zeros((1, n_points - pts.shape[1], 3))], 1)
        nors = np.concatenate([nors, np.zeros((1, n_points - nors.shape[1], 3))], 1)
        weights = np.concatenate([weights, np.zeros((1, n_points - weights.shape[1]))], 1)
    return pts, nors, weights


def generate_new_samples(sess, in_pos, in_dir, in_normal, mesh, config, scatter_pred, feature_statistics,
                         n_samples, albedo, sigma_t, g, eta, constraint_kdtree=None,
                         disable_shape_features=False, rotate_poly=True, use_legacy_kernel=False,
                         kdtree_threshold=vae.global_config.FIT_KDTREE_THRESHOLD, fit_regularization=FIT_REGULARIZATION,
                         scene=None, use_hard_constraint=True, features=None):

    assert config.dim == 3
    vae_out_pos_gen = scatter_pred.model_outputs['out_pos_gen']
    in_pos = np.atleast_2d(in_pos)
    in_dir = in_dir / np.sqrt(np.sum(in_dir ** 2))

    # Extract local geometry features
    if not features:
        features = extract_shape_features(config, in_pos, scene, constraint_kdtree, in_dir, g, sigma_t,
                                          albedo, rotate_poly, mesh, in_normal,
                                          use_legacy_kernel, kdtree_threshold, fit_regularization, use_hard_constraint)

    # print(f"features['coeffs']: {features['coeffs']}")
    samples = []
    samples2 = []
    batch_size = 32

    batch_size = min(n_samples, 1024)
    for i in range(n_samples // batch_size):
        new_z = np.random.randn(batch_size, config.n_latent)
        in_pos_batch = np.ones((batch_size, config.dim)) * in_pos
        in_normal_batch = np.ones((batch_size, config.dim)) * in_normal
        in_dir_batch = np.ones((batch_size, config.dim)) * in_dir
        points_batch, shape_features_batch = None, None
        point_normals_batch, point_weights_batch = None, None
        if config.shape_features_name:
            num_poly_coeffs = shape_feat_name_to_num_coeff(config.shape_features_name, config.dim)
            shape_features_batch = np.ones((batch_size, num_poly_coeffs)) * np.atleast_2d(features['coeffs'])
            if disable_shape_features:
                shape_features_batch *= 0.0
        elif config.use_point_net:
            points_batch = np.tile(features['points'], (batch_size, 1, 1))
            point_normals_batch = np.tile(features['point_normals'], (batch_size, 1, 1))
            point_weights_batch = np.tile(features['point_weights'], (batch_size, 1))

        adjusted_in_dir_batch = np.ones((batch_size, config.dim)) * features['adjusted_in_dir']
        poly_normal_batch = np.ones((batch_size, config.dim)) * features['poly_normal']

        albedo_batch = np.ones((batch_size, 3)) * albedo
        sigma_t_batch = np.ones((batch_size, 3)) * sigma_t
        g_batch = np.ones((batch_size, 1)) * g
        eta_batch = np.ones((batch_size, 1)) * eta
        feed_dict = generate_new_samples_feed_dict(scatter_pred.ph_manager, in_pos_batch, in_normal_batch,
                                                   in_dir_batch, shape_features_batch,
                                                   albedo_batch, sigma_t_batch, g_batch, eta_batch, feature_statistics, config,
                                                   points=points_batch, point_normals=point_normals_batch,
                                                   point_weights=point_weights_batch, adjusted_in_dir=adjusted_in_dir_batch, poly_normal=poly_normal_batch)
        feed_dict[scatter_pred.ph_manager.latent_z] = new_z
        generated_samples = sess.run(vae_out_pos_gen, feed_dict=feed_dict)
        if not config.use_similarity_theory:
            generated_samples = (generated_samples - in_pos) / sigma_t + in_pos
            if 'out_pos_gen_unproj' in scatter_pred.model_outputs.keys():
                generated_samples_unproj = sess.run(
                    scatter_pred.model_outputs['out_pos_gen_unproj'], feed_dict=feed_dict)
                generated_samples_unproj = (generated_samples_unproj - in_pos) / sigma_t + in_pos
                samples2.append(generated_samples_unproj)
        samples.append(generated_samples)
    generated_samples = np.concatenate(samples, 0)
    if len(samples2) > 0:
        generated_samples2 = np.concatenate(samples2, 0)
    else:
        generated_samples2 = generated_samples

    if config.shape_features_name:
        return generated_samples, generated_samples2, features
    return generated_samples, generated_samples, features


def vae_reconstruct_samples(sess, out_pos, coeffs, in_pos, in_dir, in_normal, config, scatter_pred, feature_statistics,
                            albedo, sigma_t, g, eta,
                            disable_shape_features=False):
    n_samples = out_pos.shape[0]
    in_pos = np.atleast_2d(in_pos)
    in_dir = in_dir / np.sqrt(np.sum(in_dir ** 2))
    # Try generating new samples
    samples = []
    batch_size = 32
    num_poly_coeffs = shape_feat_name_to_num_coeff(config.shape_features_name, config.dim)
    batch_size = min(n_samples, 1024)
    for i in range(n_samples // batch_size):
        out_pos_batch = out_pos[(i * batch_size):((i + 1) * batch_size), :]
        in_pos_batch = np.ones((batch_size, config.dim)) * in_pos
        in_normal_batch = np.ones((batch_size, config.dim)) * in_normal
        in_dir_batch = np.ones((batch_size, config.dim)) * in_dir

        points_batch, shape_features_batch = None, None
        point_normals_batch, point_weights_batch = None, None
        if config.shape_features_name:
            shape_features_batch = np.ones((batch_size, num_poly_coeffs)) * coeffs
            if disable_shape_features:
                shape_features_batch *= 0.0
        elif config.use_point_net:
            points_batch = np.tile(features['points'], (batch_size, 1, 1))
            point_normals_batch = np.tile(features['point_normals'], (batch_size, 1, 1))
            point_weights_batch = np.tile(features['point_weights'], (batch_size, 1))

        albedo_batch = np.ones((batch_size, 3)) * albedo
        sigma_t_batch = np.ones((batch_size, 3)) * sigma_t
        g_batch = np.ones((batch_size, 1)) * g
        eta_batch = np.ones((batch_size, 1)) * eta
        feed_dict = generate_new_samples_feed_dict(scatter_pred.ph_manager, in_pos_batch, in_normal_batch,
                                                   in_dir_batch, shape_features_batch,
                                                   albedo_batch, sigma_t_batch, g_batch, eta_batch, feature_statistics, config,
                                                   points=points_batch, point_normals=point_normals_batch,
                                                   point_weights=point_weights_batch, out_pos=out_pos_batch)
        generated_samples = sess.run(scatter_pred.model_outputs['out_pos'], feed_dict=feed_dict)
        if not config.scale_point_by_poly_scale and not config.use_similarity_theory:
            generated_samples = (generated_samples - in_pos) / sigma_t + in_pos
        samples.append(generated_samples)
    generated_samples = np.concatenate(samples, 0)

    return generated_samples


def estimate_absorption(sess, in_pos, in_dir, in_normal, config, absorption_pred, feature_statistics,
                        albedo, sigma_t, g, eta, features):

    assert config.dim == 3
    absorption_estimator = absorption_pred.model_outputs['absorption']
    in_pos = np.atleast_2d(in_pos)
    in_dir = in_dir / np.sqrt(np.sum(in_dir ** 2))
    num_poly_coeffs = shape_feat_name_to_num_coeff(config.shape_features_name, config.dim)
    in_pos_batch = np.ones((1, config.dim)) * in_pos
    in_normal_batch = np.ones((1, config.dim)) * in_normal
    in_dir_batch = np.ones((1, config.dim)) * in_dir
    points_batch, shape_features_batch = None, None
    point_normals_batch, point_weights_batch = None, None
    if config.shape_features_name:
        shape_features_batch = np.ones((1, num_poly_coeffs)) * features['coeffs']
    elif config.use_point_net:
        points_batch = np.tile(features['points'], (1, 1, 1))
        point_normals_batch = np.tile(features['point_normals'], (1, 1, 1))
        point_weights_batch = np.tile(features['point_weights'], (1, 1))

    albedo_batch = np.ones((1, 3)) * albedo
    sigma_t_batch = np.ones((1, 3)) * sigma_t
    g_batch = np.ones((1, 1)) * g
    eta_batch = np.ones((1, 1)) * eta

    feed_dict = generate_new_samples_feed_dict(absorption_pred.ph_manager, in_pos_batch, in_normal_batch,
                                               in_dir_batch, shape_features_batch,
                                               albedo_batch, sigma_t_batch, g_batch, eta_batch, feature_statistics, config,
                                               points=points_batch, point_normals=point_normals_batch,
                                               point_weights=point_weights_batch)

    absorption = sess.run(absorption_estimator, feed_dict=feed_dict)
    return absorption


def sample_outgoing_directions(sess, in_pos, in_dir, in_normal, out_pos, out_normals, config, angular_pred, feature_statistics,
                               n_samples, dataset, albedo, sigma_t, g, eta, coeffs, disable_shape_features=False):

    assert config.dim == 3
    in_pos = np.atleast_2d(in_pos)
    in_dir = in_dir / np.sqrt(np.sum(in_dir ** 2))

    # Try generating new samples
    samples = []
    batch_size = 32
    num_poly_coeffs = shape_feat_name_to_num_coeff(config.shape_features_name, config.dim)

    batch_size = min(n_samples, 1024)

    batch_sizes = [batch_size] * (n_samples // batch_size)
    if n_samples % batch_size != 0:
        batch_sizes.append(n_samples - (n_samples // batch_size) * batch_size)

    n = 0
    for i, batch_size in enumerate(batch_sizes):
        if config.use_vmf:
            new_z = np.random.rand(batch_size, 2)
        else:
            new_z = np.random.randn(batch_size, config.n_latent)
        in_pos_batch = np.ones((batch_size, config.dim)) * in_pos
        in_normal_batch = np.ones((batch_size, config.dim)) * in_normal
        in_dir_batch = np.ones((batch_size, config.dim)) * in_dir
        if config.shape_features_name:
            shape_features_batch = np.ones((batch_size, num_poly_coeffs)) * coeffs
            if disable_shape_features:
                shape_features_batch *= 0.0
        else:
            shape_features_batch = None

        albedo_batch = np.ones((batch_size, 3)) * albedo
        sigma_t_batch = np.ones((batch_size, 3)) * sigma_t
        g_batch = np.ones((batch_size, 1)) * g
        eta_batch = np.ones((batch_size, 1)) * eta

        feed_dict = generate_new_samples_feed_dict(angular_pred.ph_manager, in_pos_batch, in_normal_batch,
                                                   in_dir_batch, shape_features_batch,
                                                   albedo_batch, sigma_t_batch, g_batch, eta_batch, feature_statistics, config,
                                                   out_pos=out_pos[n:n + batch_size, :],
                                                   out_normals=out_normals[n:n + batch_size, :])

        feed_dict[angular_pred.ph_manager.angular_latent_z] = new_z
        # generated_samples = sess.run(angular_pred.model_outputs['out_dir_gen'], feed_dict=feed_dict)
        generated_samples = sess.run(angular_pred.model_outputs['vmf_dir_ws'], feed_dict=feed_dict)
        gen_k, gen_dir = sess.run([angular_pred.model_outputs['vmf_k'],
                                   angular_pred.model_outputs['vmf_dir']], feed_dict=feed_dict)
        generated_samples = utils.math.normalize(generated_samples)
        samples.append(generated_samples)
        n += generated_samples.shape[0]
    generated_samples = np.concatenate(samples, 0)
    return generated_samples
