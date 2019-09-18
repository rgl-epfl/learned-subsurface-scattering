import os
from itertools import chain

import numpy as np
import tensorflow as tf
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon

import vae.datahandler
import vae.utils
from vae import tf_utils
from vae.utils import (extract_poly_order_from_feat_name, get_feature_stats,
                       shape_feat_name_to_num_coeff)

import utils.transforms

def batch_to_feed_dict(predictor, batch, feature_statistics, config, is_training=True):
    indices = vae.datahandler.batch_indices(config)

    feed_dict = {}
    feed_dict[predictor.in_pos_p] = batch[indices['inPos']]
    feed_dict[predictor.out_pos_p] = batch[indices['outPos']]

    feed_dict[predictor.in_normal_p] = batch[indices['inNormal']]
    feed_dict[predictor.in_dir_p] = batch[indices['inDir']]
    feed_dict[predictor.out_normal_p] = batch[indices['outNormal']]
    feed_dict[predictor.out_dir_p] = batch[indices['outDir']]

    feed_dict[predictor.phase_p] = is_training

    if config.shape_features_name:
        feed_dict[predictor.shape_features_p] = batch[indices[config.shape_features_name]]
        f = config.shape_features_name
        feed_dict[predictor.shape_features_mean_p] = feature_statistics['{}_mean'.format(f)]
        feed_dict[predictor.shape_features_stdinv_p] = feature_statistics['{}_stdinv'.format(f)]

        g = batch[indices['g']]
        albedo = batch[indices['albedo']]
        sigma_t = batch[indices['sigmaT']]
        feed_dict[predictor.poly_scale_factor_p] = vae.utils.get_poly_scale_factor(vae.utils.kernel_epsilon(g, sigma_t, albedo))

    if config.use_outpos_statistics:
        feed_dict[predictor.out_pos_mean_p] = feature_statistics['outPosRel{}_mean'.format(config.prediction_space)]
        feed_dict[predictor.out_pos_stdinv_p] = feature_statistics['outPosRel{}_stdinv'.format(config.prediction_space)]


    feed_dict[predictor.azimuth_transf_p] = utils.transforms.to_azimuth_space(-batch[indices['inDir']], batch[indices['inNormal']])


    feed_dict[predictor.absorption_prob_p] = batch[indices['absorptionProb']]

    feed_dict[predictor.albedo_p] = batch[indices['albedo']]
    feed_dict[predictor.albedo_mean_p] = feature_statistics['albedo_mean']
    feed_dict[predictor.albedo_stdinv_p] = feature_statistics['albedo_stdinv']
    feed_dict[predictor.eff_albedo_p] = batch[indices['effAlbedo']]
    feed_dict[predictor.eff_albedo_mean_p] = feature_statistics['effAlbedo_mean']
    feed_dict[predictor.eff_albedo_stdinv_p] = feature_statistics['effAlbedo_stdinv']

    feed_dict[predictor.g_p] = batch[indices['g']]
    feed_dict[predictor.g_mean_p] = feature_statistics['g_mean']
    feed_dict[predictor.g_stdinv_p] = feature_statistics['g_stdinv']
    feed_dict[predictor.sigma_t_p] = batch[indices['sigmaT']]
    feed_dict[predictor.sigma_t_mean_p] = feature_statistics['sigmaT_mean']
    feed_dict[predictor.sigma_t_stdinv_p] = feature_statistics['sigmaT_mean']
    feed_dict[predictor.ior_p] = batch[indices['ior']]
    feed_dict[predictor.dropout_keep_prob_p] = config.dropout_keep_prob if is_training else 1.0
    return feed_dict


def compute_learningrate(learningrate, global_step, config):
    if config.use_adaptive_lr:
        init_lr = learningrate
        min_lr = init_lr / 8.0
        decay_rate = 0.8
        learningrate = tf.clip_by_value(
            tf.train.exponential_decay(init_lr, global_step, 100000,
                                       decay_rate, staircase=True), min_lr, 1)
    return learningrate


def create_optimizer(config, learningrate, loss, global_step):
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        # Ensures that we execute the update_ops before performing the train_step
        # Update steps are used for batch norm
        if config.optimizer == 'adam':
            optimizer = tf.train.AdamOptimizer(learningrate)
        elif config.optimizer == 'sgd':
            optimizer = tf.train.GradientDescentOptimizer(learningrate)
        elif config.optimizer == 'adadelta':
            optimizer = tf.train.AdadeltaOptimizer(learningrate)
        elif config.optimizer == 'adagrad':
            optimizer = tf.train.AdagradOptimizer(learningrate)
        elif config.optimizer == 'rmsprop':
            optimizer = tf.train.RMSPropOptimizer(learningrate)
        else:
            raise ValueError('Invalid optimizer {}'.format(config.optimizer))

        # Clip gradients
        grads_and_vars = optimizer.compute_gradients(loss, tf.trainable_variables())
        grads, vars_ = zip(*grads_and_vars)
        capped_grads, gradient_norm = tf.clip_by_global_norm(grads, clip_norm=config.clip_gradient_threshold)
        gradient_norm = tf.check_numerics(gradient_norm, "Gradient norm is NaN or Inf.")
        train_step = optimizer.apply_gradients(zip(capped_grads, vars_), global_step=global_step)
        return train_step


def train(train_data_iterator, test_data_iterator,
          learningrate, feature_statistics, logdir, config,
          restore, ncores, predictors):

    global_step = tf.Variable(0, name='global_step', trainable=False)
    losses = predictors[0].get_losses(global_step)

    for i in range(1, len(predictors)):
        other_losses = predictors[i].get_losses(global_step)
        for k in other_losses.keys():
            new_k = predictors[i].prefix + '/' + k
            losses[new_k] = other_losses[k]
        losses['loss'] += other_losses['loss'] * predictors[i].config.loss_weight

    # Add potential regularization losses to final loss
    losses['loss'] += tf.losses.get_regularization_loss()

    learningrate = compute_learningrate(learningrate, global_step, config)
    train_step = create_optimizer(config, learningrate, losses['loss'],  global_step)

    saver = tf.train.Saver(save_relative_paths=True)
    summary_op = tf.summary.merge_all()

    sess_config = tf.ConfigProto(intra_op_parallelism_threads=ncores, inter_op_parallelism_threads=ncores,
                                 allow_soft_placement=True, device_count={'CPU': ncores})
    with tf.Session(config=sess_config) as sess:
        sess.run(tf.global_variables_initializer())
        if restore:
            vae.tf_utils.restore_model(sess, logdir)
            print("Model restore finished, current global step: {}".format(global_step .eval()))
        test_summary_writer = tf.summary.FileWriter(os.path.join(logdir, 'test'), sess.graph)
        train_summary_writer = tf.summary.FileWriter(os.path.join(logdir, 'train'), sess.graph)
        save_path = saver.save(sess, os.path.join(logdir, 'model.ckpt'), global_step=global_step)
        print('Untrained model saved in file: {}'.format(save_path))

        while True:
            try:
                batch = sess.run(train_data_iterator)
                train_feed_dict = batch_to_feed_dict(predictors[0].ph_manager, batch, feature_statistics, config)
                train_step.run(feed_dict=train_feed_dict)
                step = sess.run(global_step)
                if step % 5000 == 0 or step == 50:
                    print('Evaluating test loss...')
                    test_loss_values = []
                    for _ in losses:
                        test_loss_values.append([])

                    output_tensors = [losses[k] for k in losses]
                    for _ in range(10):
                        test_feed_dict = batch_to_feed_dict(predictors[0].ph_manager, sess.run(
                            test_data_iterator), feature_statistics, config, is_training=False)
                        loss_results = sess.run(output_tensors, test_feed_dict)
                        for j, v in enumerate(loss_results):
                            test_loss_values[j].append(v)
                    for i, v in enumerate(test_loss_values):
                        test_loss_values[i] = np.mean(np.concatenate([np.ravel(x) for x in v]))

                    summary = tf.Summary()
                    for i, v in enumerate(losses):
                        summary.value.add(tag=f"loss/{v}", simple_value=test_loss_values[i])

                    test_summary_writer.add_summary(summary, step)
                    res = sess.run(output_tensors + [summary_op], train_feed_dict)
                    summary_train = res[-1]
                    train_loss_string = '\t training'
                    test_loss_string = '\t test'
                    for i, v in enumerate(losses):
                        train_loss_string += f' {v}: {np.mean(res[i]):.4f}'
                        test_loss_string += f' {v}: {test_loss_values[i]:.4f}'
                    print(f'step {step} \n{train_loss_string} \n{test_loss_string}')
                    train_summary_writer.add_summary(summary_train, step)
                if step % 10000 == 0:
                    save_path = saver.save(sess, os.path.join(logdir, 'model.ckpt'), global_step=global_step)
                    print('Model saved in file: {}'.format(save_path))

            except tf.errors.OutOfRangeError:
                break

        train_summary_writer.close()
