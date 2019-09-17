import os

import numpy as np
import tensorflow as tf

from utils.printing import printg


class Trainer:

    def __init__(self, patch_size):
        self.surf_area = tf.placeholder(
            tf.float32, shape=[None, patch_size[0], patch_size[1], patch_size[2], 1], name='surf_area')
        self.interior = tf.placeholder(
            tf.float32, shape=[None, patch_size[0], patch_size[1], patch_size[2], 1], name='interior')
        self.voxel_normals = tf.placeholder(
            tf.float32, shape=[None, patch_size[0], patch_size[1], patch_size[2], 3], name='voxel_normals')
        self.voxel_t1 = tf.placeholder(
            tf.float32, shape=[None, patch_size[0], patch_size[1], patch_size[2], 3], name='voxel_t1')
        self.voxel_t2 = tf.placeholder(
            tf.float32, shape=[None, patch_size[0], patch_size[1], patch_size[2], 3], name='voxel_t2')
        self.vxbb_min = tf.placeholder(tf.float32, shape=[None, 3], name='vxbb_min')
        self.vxbb_max = tf.placeholder(tf.float32, shape=[None, 3], name='vxbb_max')
        self.scattering = tf.placeholder(
            tf.float32, shape=[None, patch_size[0], patch_size[1], patch_size[2], 1], name='ref_scattering')
        self.scene_idx = tf.placeholder(tf.int32, shape=[None, 1], name='scene_idx')
        self.normal = tf.placeholder(tf.float32, shape=[None, 3], name='normal')
        self.tangent1 = tf.placeholder(tf.float32, shape=[None, 3], name='tangent1')
        self.tangent2 = tf.placeholder(tf.float32, shape=[None, 3], name='tangent2')
        self.position = tf.placeholder(tf.float32, shape=[None, 3], name='position')

    def restore_model(self, session, logdir):
        saver = tf.train.Saver()
        saver.restore(session, tf.train.latest_checkpoint(logdir))

    def pred(self, model_output, distrib_projector):
        return distrib_projector.tf_project(model_output, scene_idx=self.scene_idx, surf_area=self.surf_area,
                                            voxel_normals=self.voxel_normals, voxel_t1=self.voxel_t1, voxel_t2=self.voxel_t2, vxbb_min=self.vxbb_min, vxbb_max=self.vxbb_max)

    def batch_to_feed_dict(self, batch):
        return {self.scene_idx: batch[0],
                self.surf_area: batch[1],
                self.interior: batch[2],
                self.voxel_normals: batch[3],
                self.voxel_t1: batch[11],
                self.voxel_t2: batch[12],
                self.scattering: batch[4],
                self.vxbb_min: batch[5],
                self.vxbb_max: batch[6],
                self.normal: batch[7],
                self.tangent1: batch[8],
                self.tangent2: batch[9],
                self.position: batch[10]}

    def train(self, data_set_iterator, test_set_iterator, prediction, model_output, nn_output_relative, learningrate, logdir, restore=False):

        # map output params to actual distrib and compute loss
        with tf.name_scope('loss'):
            loss = tf.reduce_mean(tf.square(prediction - self.scattering[:, :, :, :, 0]))
            tf.summary.scalar('loss', loss)

        global_step = tf.Variable(0, name='global_step', trainable=False)

        init_lr = learningrate
        min_lr = init_lr / 8.0
        decay_rate = 0.8
        learningrate = tf.clip_by_value(
            tf.train.exponential_decay(init_lr, global_step, 100000,
                                       decay_rate, staircase=True), min_lr, 1)

        train_step = tf.train.AdamOptimizer(learningrate).minimize(loss, global_step=global_step)
        saver = tf.train.Saver()
        summary_op = tf.summary.merge_all()

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            if restore:
                self.restore_model(sess, logdir)
                print("Model restore finished, current global step: {}".format(global_step.eval()))
            train_summary_writer = tf.summary.FileWriter(os.path.join(logdir, 'train'), sess.graph)
            test_summary_writer = tf.summary.FileWriter(os.path.join(logdir, 'test'), sess.graph)

            save_path = saver.save(sess, os.path.join(logdir, 'model.ckpt'), global_step=global_step)
            print('Untrained model saved in file: {}'.format(save_path))

            while True:
                try:
                    train_feed_dict = self.batch_to_feed_dict(sess.run(data_set_iterator))
                    # print('sess.run(nn_output_relative): {}'.format(sess.run(nn_output_relative, feed_dict=train_feed_dict)[0]))
                    train_step.run(feed_dict=train_feed_dict)
                    step = sess.run(global_step)
                    # print('sess.run(nn_output_relative) new: {}'.format(sess.run(nn_output_relative, feed_dict=train_feed_dict)[0]))

                    if step % 500 == 0:
                        test_accuracies = []
                        for i in range(10):
                            test_feed_dict = self.batch_to_feed_dict(sess.run(test_set_iterator))
                            test_loss_value, rel_pred_params = sess.run([loss, nn_output_relative], test_feed_dict)
                            test_accuracies.append(test_loss_value)
                        avg_test_accuracy = np.mean(test_accuracies)
                        print('rel_pred_params[0]: {}'.format(rel_pred_params[0]))
                        # Create a new Summary object with your measure
                        summary = tf.Summary()
                        summary.value.add(tag="loss/loss", simple_value=avg_test_accuracy)
                        test_summary_writer.add_summary(summary, step)

                        train_loss_value, summary_train = sess.run([loss, summary_op], train_feed_dict)

                        print('step {}, training loss {} test loss {}'.format(step, train_loss_value, test_loss_value))
                        # test_summary_writer.add_summary(summary_test, step)
                        train_summary_writer.add_summary(summary_train, step)
                    if step % 5000 == 0:
                        save_path = saver.save(sess, os.path.join(logdir, 'model.ckpt'), global_step=global_step)
                        print('Model saved in file: {}'.format(save_path))

                except tf.errors.OutOfRangeError:
                    break

            train_summary_writer.close()
            test_summary_writer.close()
