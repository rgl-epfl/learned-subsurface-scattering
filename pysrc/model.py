import numpy as np
import tensorflow as tf

from utils.printing import *
from scattering_distributions import *

from datahandler import load_scenes

def weight_variable(shape):
    # TODO: Potentially can also pass seed here if global seed is not enough
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))


def bias_variable(shape):
    return tf.Variable(tf.constant(0.05, shape=shape))


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def conv3d(x, W):
    return tf.nn.conv3d(x, W, strides=[1, 1, 1, 1, 1], padding='SAME')


def deconv2d(x, W, stride):
    x_shape = tf.shape(x)  # get dynamic batch size
    output_shape = tf.stack([x_shape[0], x_shape[1] * 2, x_shape[2] * 2, x_shape[3] // 2])
    return tf.nn.conv2d_transpose(x, W, output_shape, strides=[1, stride, stride, 1],
                                  padding='SAME')


def max_pool_2x2x2(x):
    return tf.nn.max_pool3d(x, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1],
                            padding='SAME')


def upsample_2(x, kernel_size):
    W = weight_variable([kernel_size, kernel_size, x.shape[3].value // 2, x.shape[3].value])
    b = bias_variable([x.shape[3].value // 2])
    h = tf.nn.relu(deconv2d(x, W, 2) + b)
    return h


def conv2d_layer(x, input_channels, output_channels, kernel_size, use_relu=True):
    W = weight_variable([kernel_size, kernel_size, input_channels, output_channels])
    b = bias_variable([output_channels])
    if use_relu:
        h = tf.nn.relu(conv2d(x, W) + b)
    else:
        h = conv2d(x, W) + b
    return h


def conv3d_layer(x, input_channels, output_channels, kernel_size, use_relu=True):
    W = weight_variable([kernel_size, kernel_size, kernel_size, input_channels, output_channels])
    b = bias_variable([output_channels])
    if use_relu:
        h = tf.nn.relu(conv3d(x, W) + b)
    else:
        h = conv3d(x, W) + b
    return h


def remap_variable(x, center, min_offset, max_offset):
    rel_offset = (max_offset - min_offset) * x + min_offset
    return rel_offset + center, rel_offset


def get_position_prediction(trainer, nn_output, use_tangent_coordinates=False):

    # TODO: The 0.1 here should be depending on the mean free path of the medium
    pos_center = trainer.position - 0.1 * trainer.normal
    if use_tangent_coordinates:
        pos_max_d = np.array([0.5, 0.5, 0.5])
        pos_min_d = np.array([-0.5, -0.5, -0.5])
    else:
        pos_max_d = np.array([0.5, 0.5, 0.5])
        pos_min_d = np.array([-0.5, -0.5, -0.5])

    pos_local_relative = nn_output * (pos_max_d - pos_min_d) + pos_min_d

    if use_tangent_coordinates:
        posX = pos_local_relative[:, 1] * trainer.normal[:, 0] + pos_local_relative[:, 2] * \
            trainer.tangent1[:, 0] + pos_local_relative[:, 3] * trainer.tangent2[:, 0]
        posY = pos_local_relative[:, 1] * trainer.normal[:, 1] + pos_local_relative[:, 2] * \
            trainer.tangent1[:, 1] + pos_local_relative[:, 3] * trainer.tangent2[:, 1]
        posZ = pos_local_relative[:, 1] * trainer.normal[:, 2] + pos_local_relative[:, 2] * \
            trainer.tangent1[:, 2] + pos_local_relative[:, 3] * trainer.tangent2[:, 2]
        position_ws = tf.stack([posX, posY, posZ], axis=1, name='pred_position_rel') + pos_center
    else:
        position_ws = pos_local_relative + pos_center

    return position_ws, pos_local_relative


def one_layer_pointlight(trainer, args, config):
    """Naive 1 layer model with no input features. Can be used to debug convergence for '0-prediction-dataset'"""

    with tf.name_scope('input') as scope:
        in_data = tf.concat([tf.ones_like(trainer.interior)], axis=4)
    with tf.name_scope('pool') as scope:
        pool1 = max_pool_2x2x2(in_data)

    vectorized = tf.reshape(pool1,
                            [-1, pool1.shape[1] * pool1.shape[2] * pool1.shape[3] * pool1.shape[4]])

    with tf.name_scope('FC2') as scope:
        nn_output_raw = tf.contrib.slim.fully_connected(
            vectorized, 7, activation_fn=None, weights_initializer=tf.random_normal_initializer(stddev=0.1))

    with tf.name_scope('clamping') as scope:
        # Remap output of neural network to sensible ranges
        nn_output = tf.sigmoid(nn_output_raw)
        position_pred, position_relative = get_position_prediction(trainer, nn_output[:, 1:4])
        sigma_t_pred = tf.constant(0.0)  # Dont use sigma_t here

        pred_params = tf.stack([tf.expand_dims(sigma_t_pred, 1), position_pred], axis=1)
        pred_params_relative = tf.stack([tf.expand_dims(tf.constant(0.0), 1), position_relative], axis=1)

    distrib_projector = PointLightProjector()
    prediction = distrib_projector.tf_project(pred_params, scene_idx=trainer.scene_idx, surf_area=trainer.surf_area,
                                              voxel_normals=trainer.voxel_normals, voxel_t1=trainer.voxel_t1,
                                              voxel_t2=trainer.voxel_t2, vxbb_min=trainer.vxbb_min, vxbb_max=trainer.vxbb_max)
    return prediction, pred_params, pred_params_relative, nn_output_raw


def baseline_pointlight(trainer, args, config):
    """Sets up a simple 3D-CNN to predict scattering parameters"""

    feature_dims = [1, 1, 3]
    input_features = [trainer.surf_area, trainer.interior, trainer.voxel_normals]

    # DEBUG: Simple network, using only 1 feature
    feature_dims = [1]
    input_features = [trainer.interior]

    # Concatenate features
    with tf.name_scope('input') as scope:
        in_data = tf.concat(input_features, axis=4)

    with tf.name_scope('layer1') as scope:
        conv2 = conv3d_layer(in_data, np.sum(feature_dims), 4, 3, True)  # 16x16x16 input
        pool1 = max_pool_2x2x2(conv2)
    with tf.name_scope('layer2') as scope:
        conv3 = conv3d_layer(pool1, 4, 8, 3, True)
        pool2 = max_pool_2x2x2(conv3)
    with tf.name_scope('layer3') as scope:
        conv3 = conv3d_layer(pool2, 8, 16, 3, True)
        pool3 = max_pool_2x2x2(conv3)

    # Vectorize and use FC layers to predict distrib parameters
    vectorized = tf.reshape(pool3,
                            [-1, pool3.shape[1] * pool3.shape[2] * pool3.shape[3] * pool3.shape[4]])

    if config.use_tangent_coordinates:
        vectorized = tf.concat([vectorized, trainer.tangent1, trainer.tangent2], axis=1)

    with tf.name_scope('FC1') as scope:
        fc1 = tf.contrib.slim.fully_connected(vectorized, 512)

    if config.use_tangent_coordinates:
        fc1 = tf.concat([fc1, trainer.tangent1, trainer.tangent2], axis=1)

    # Final fully connected layer WITHOUT ReLU activation
    with tf.name_scope('FC2') as scope:
        nn_output_raw = tf.contrib.slim.fully_connected(fc1, 6, activation_fn=None)

    with tf.name_scope('clamping') as scope:
        # Remap output of neural network to sensible ranges
        nn_output = tf.sigmoid(nn_output_raw)
        position_pred, position_relative = get_position_prediction(trainer, nn_output[:, 1:4], config.use_tangent_coordinates)
        sigma_t_pred, sigma_t_relative = remap_variable(nn_output[:, 0], 0.0, 0.0, 0.0)  # Dont use sigma_t here
        pred_params = tf.concat([tf.expand_dims(sigma_t_pred, 1), position_pred], axis=1)
        pred_params_relative = tf.concat([tf.expand_dims(sigma_t_relative, 1), position_relative], axis=1)

    distrib_projector = PointLightProjector()
    prediction = distrib_projector.tf_project(pred_params, scene_idx=trainer.scene_idx, surf_area=trainer.surf_area,
                                              voxel_normals=trainer.voxel_normals, voxel_t1=trainer.voxel_t1,
                                              voxel_t2=trainer.voxel_t2, vxbb_min=trainer.vxbb_min, vxbb_max=trainer.vxbb_max)

    return prediction, pred_params, pred_params_relative, nn_output_raw


def sum_of_gaussian_pointlight(trainer, args, config):
    # DEBUG: Simple network, using only 1 feature
    feature_dims = [1]
    input_features = [trainer.interior]

    # Concatenate features
    with tf.name_scope('input') as scope:
        in_data = tf.concat(input_features, axis=4)

    with tf.name_scope('layer1') as scope:
        conv2 = conv3d_layer(in_data, np.sum(feature_dims), 4, 3, True) # 16x16x16 input
        pool1 = max_pool_2x2x2(conv2)
    with tf.name_scope('layer2') as scope:
        conv3 = conv3d_layer(pool1, 4, 8, 3, True)
        pool2 = max_pool_2x2x2(conv3)
    with tf.name_scope('layer3') as scope:
        conv3 = conv3d_layer(pool2, 8, 16, 3, True)
        pool3 = max_pool_2x2x2(conv3)

    # Vectorize and use FC layers to predict distrib parameters
    vectorized = tf.reshape(pool3,
                            [-1, pool3.shape[1] * pool3.shape[2] * pool3.shape[3] * pool3.shape[4]])

    if config.use_tangent_coordinates:
        vectorized = tf.concat([vectorized, trainer.tangent1, trainer.tangent2], axis=1)

    with tf.name_scope('FC1') as scope:
        fc1 = tf.contrib.slim.fully_connected(vectorized, 512)

    if config.use_tangent_coordinates:
        fc1 = tf.concat([fc1, trainer.tangent1, trainer.tangent2], axis=1)

    # Final fully connected layer WITHOUT ReLU activation
    with tf.name_scope('FC2') as scope:
        nn_output_raw = tf.contrib.slim.fully_connected(fc1, 7, activation_fn=None)

    with tf.name_scope('clamping') as scope:
        # Remap output of neural network to sensible ranges
        nn_output = tf.sigmoid(nn_output_raw[:, 0:4])
        position_pred, position_relative = get_position_prediction(trainer, nn_output[:, 1:4], config.use_tangent_coordinates)
        sigma_t_pred, sigma_t_relative = remap_variable(nn_output[:, 0], 0.0, 0.0, 0.0)  # Dont use sigma_t here for now
        # Compute weights for sum of gaussians
        weights = tf.nn.softmax(nn_output_raw[:, 4:7])
        pred_params = tf.concat([tf.expand_dims(sigma_t_pred, 1), position_pred, weights, trainer.normal], axis=1)
        pred_params_relative = tf.concat([tf.expand_dims(sigma_t_relative, 1), position_relative, weights], axis=1)

    distrib_projector = PointLightProjector(use_spherical_gaussian=True)
    prediction = distrib_projector.tf_project(pred_params, scene_idx=trainer.scene_idx, surf_area=trainer.surf_area,
                                              voxel_normals=trainer.voxel_normals, voxel_t1=trainer.voxel_t1,
                                              voxel_t2=trainer.voxel_t2, vxbb_min=trainer.vxbb_min, vxbb_max=trainer.vxbb_max)

    return prediction, pred_params, pred_params_relative, nn_output_raw


def sum_of_gaussian_pointlight_wide(trainer, args, config):
    feature_dims = [1, 1, 3]
    input_features = [trainer.surf_area, trainer.interior, trainer.voxel_normals]

    # Concatenate features
    with tf.name_scope('input') as scope:
        in_data = tf.concat(input_features, axis=4)
    with tf.name_scope('layer1') as scope:
        conv2 = conv3d_layer(in_data, np.sum(feature_dims), 8, 3, True) # 16x16x16 input
        pool1 = max_pool_2x2x2(conv2)
    with tf.name_scope('layer2') as scope:
        conv3 = conv3d_layer(pool1, 8, 16, 3, True)
        pool2 = max_pool_2x2x2(conv3)
    with tf.name_scope('layer3') as scope:
        conv3 = conv3d_layer(pool2, 16, 32, 3, True)
        pool3 = max_pool_2x2x2(conv3)

    # Vectorize and use FC layers to predict distrib parameters
    vectorized = tf.reshape(pool3,
                            [-1, pool3.shape[1] * pool3.shape[2] * pool3.shape[3] * pool3.shape[4]])

    if config.use_tangent_coordinates:
        vectorized = tf.concat([vectorized, trainer.tangent1, trainer.tangent2], axis=1)

    with tf.name_scope('FC1') as scope:
        fc1 = tf.contrib.slim.fully_connected(vectorized, 512)

    if config.use_tangent_coordinates:
        fc1 = tf.concat([fc1, trainer.tangent1, trainer.tangent2], axis=1)

    # Final fully connected layer WITHOUT ReLU activation
    with tf.name_scope('FC2') as scope:
        nn_output_raw = tf.contrib.slim.fully_connected(fc1, 7, activation_fn=None)

    with tf.name_scope('clamping') as scope:
        # Remap output of neural network to sensible ranges
        nn_output = tf.sigmoid(nn_output_raw[:, 0:4])
        position_pred, position_relative = get_position_prediction(trainer, nn_output[:, 1:4], config.use_tangent_coordinates)
        sigma_t_pred, sigma_t_relative = remap_variable(nn_output[:, 0], 0.0, 0.0, 0.0)  # Dont use sigma_t here for now

        # Compute weights for sum of gaussians
        weights = tf.nn.softmax(nn_output_raw[:, 4:7])
        pred_params = tf.concat([tf.expand_dims(sigma_t_pred, 1), position_pred, weights, trainer.normal], axis=1)
        pred_params_relative = tf.concat([tf.expand_dims(sigma_t_relative, 1), position_relative, weights], axis=1)

    distrib_projector = PointLightProjector(use_spherical_gaussian=True)
    prediction = distrib_projector.tf_project(pred_params, scene_idx=trainer.scene_idx, surf_area=trainer.surf_area,
                                              voxel_normals=trainer.voxel_normals, voxel_t1=trainer.voxel_t1,
                                              voxel_t2=trainer.voxel_t2, vxbb_min=trainer.vxbb_min, vxbb_max=trainer.vxbb_max)
    return prediction, pred_params, pred_params_relative, nn_output_raw




def baseline_gaussian(trainer, args, config):
    """Sets up a simple 3D-CNN to predict scattering parameters"""

    traindatadir = args.traindatadir
    feature_dims = [1, 1, 3]
    input_features = [trainer.surf_area, trainer.interior, trainer.voxel_normals]

    # Concatenate features
    with tf.name_scope('input') as scope:
        in_data = tf.concat(input_features, axis=4)

    with tf.name_scope('layer1') as scope:
        conv2 = conv3d_layer(in_data, np.sum(feature_dims), 4, 3, True)  # 16x16x16 input
        pool1 = max_pool_2x2x2(conv2)
    with tf.name_scope('layer2') as scope:
        conv3 = conv3d_layer(pool1, 4, 8, 3, True)
        pool2 = max_pool_2x2x2(conv3)
    with tf.name_scope('layer3') as scope:
        conv3 = conv3d_layer(pool2, 8, 16, 3, True)
        pool3 = max_pool_2x2x2(conv3)

    # Vectorize and use FC layers to predict distrib parameters
    vectorized = tf.reshape(pool3,
                            [-1, pool3.shape[1] * pool3.shape[2] * pool3.shape[3] * pool3.shape[4]])

    if config.use_tangent_coordinates:
        vectorized = tf.concat([vectorized, trainer.tangent1, trainer.tangent2], axis=1)

    with tf.name_scope('FC1') as scope:
        fc1 = tf.contrib.slim.fully_connected(vectorized, 512)

    if config.use_tangent_coordinates:
        fc1 = tf.concat([fc1, trainer.tangent1, trainer.tangent2], axis=1)

    # Final fully connected layer WITHOUT ReLU activation
    with tf.name_scope('FC2') as scope:
        nn_output_raw = tf.contrib.slim.fully_connected(fc1, 6, activation_fn=None)

    with tf.name_scope('clamping') as scope:

        # Transform reference normal to polar coordintes
        theta_center = tf.acos(trainer.normal[:, 2])
        phi_center = tf.atan2(trainer.normal[:, 1],  trainer.normal[:, 0])
        theta_center = tf.expand_dims(theta_center, 1)
        phi_center = tf.expand_dims(phi_center, 1)

        nn_output = tf.sigmoid(nn_output_raw)

        position_pred, position_relative = get_position_prediction(trainer, nn_output[:, 1:4], config.use_tangent_coordinates)
        sigma2_pred, sigma2_relative = remap_variable(nn_output[:, 0], 0.2, -0.15, 0.3)  # Dont use sigma_t here

        sigma2_pred = tf.expand_dims(sigma2_pred, 1)
        sigma2_relative = tf.expand_dims(sigma2_relative, 1)

        theta_pred, theta_relative = remap_variable(nn_output[:, 4], theta_center, -np.pi / 2.0, np.pi / 2.0)
        phi_pred, phi_relative = remap_variable(nn_output[:, 5], phi_center, -np.pi, np.pi)

        theta_relative = tf.expand_dims(theta_relative, 1)
        phi_relative = tf.expand_dims(phi_relative, 1)

        nn_output = tf.concat([sigma2_pred, position_pred, theta_pred, phi_pred], axis=1)
        pred_params_relative = tf.concat([sigma2_relative, position_relative, theta_relative, phi_relative], axis=1)

    # map through sperical coordinate transform to obtain the actual normal
    with tf.name_scope('spherical_coordinates') as scope:
        theta = nn_output[:, 4]
        phi = nn_output[:, 5]
        sin_theta = tf.sin(theta)
        pred_params = tf.stack([nn_output[:, 0], nn_output[:, 1], nn_output[:, 2], nn_output[:, 3],
                                sin_theta * tf.cos(phi), sin_theta * tf.sin(phi), tf.cos(theta)], axis=1, name='predParams')

    printr('create projectr')

    distrib_projector = GaussianProjector(load_scenes(traindatadir))
    printr('create projectr 22')

    prediction = distrib_projector.tf_project(pred_params, scene_idx=trainer.scene_idx, surf_area=trainer.surf_area,
                                              voxel_normals=trainer.voxel_normals, voxel_t1=trainer.voxel_t1,
                                              voxel_t2=trainer.voxel_t2, vxbb_min=trainer.vxbb_min, vxbb_max=trainer.vxbb_max)

    printr('done ctor model')
    return prediction, pred_params, pred_params_relative, nn_output_raw
