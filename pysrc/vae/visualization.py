import glob
import pickle

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from matplotlib.patches import Polygon
from PIL import Image, ImageDraw

import mitsuba.core
import mitsuba.render
import vae.config
import vae.utils
from vae.utils import get_feature_stats


def get_groundtruth_histogram(path_t, albedo, sigma_t, relative_in_angle, poly_idx, scene_dir, traindata_file, generate=False):
    polygons = vae.utils.load_polygons(scene_dir)

    pos = polygons[poly_idx]
    ref_pos, _, _ = vae.utils.index_polygon(path_t, pos)
    _, inNormal = vae.utils.project_points_on_poly(pos, np.atleast_2d(ref_pos), True)
    inDirection = vae.utils.relative_angle_to_direction(relative_in_angle, inNormal)
    if not generate:
        with open(traindata_file + '.pickle', 'rb') as f:
            train_data = pickle.load(f)

        thresh = 0.00085
        indices = train_data['polygonIdx'][:, 0] == [poly_idx]

        for k in train_data.keys():
            train_data[k] = train_data[k][indices, :]

        indices = ((train_data['inPos'][:, 0] - ref_pos[0]) ** 2 < thresh) * \
            ((train_data['inPos'][:, 1] - ref_pos[1]) ** 2 < thresh) \
            * ((train_data['inDir'][:, 0] + inDirection[0]) ** 2 < thresh * 4) * \
            ((train_data['inDir'][:, 1] + inDirection[1]) ** 2 < thresh * 4)

        for k in train_data.keys():
            train_data[k] = train_data[k][indices, :]

        outPos = train_data['outPos']
    else:
        ref_pos_mts = mitsuba.core.Point2(ref_pos[0], ref_pos[1])
        ref_dir_mts = mitsuba.core.Vector2(-inDirection[0], -inDirection[1])
        tmp_result = mitsuba.render.Volpath2D.sampleFixedStart([mitsuba.core.Point2(r[0], r[1]) for r in pos],
                                                               [mitsuba.core.Spectrum(albedo)], [mitsuba.core.Spectrum(sigma_t)], 1024, True, 128, ref_pos_mts, ref_dir_mts)

        outPos = tmp_result['outPos']
        outPosNp = []
        for i in range(len(outPos)):
            outPosNp.append(np.array([outPos[i].x, outPos[i].y]))
        outPos = np.array(outPosNp)

    hist, _, _ = np.histogram2d(outPos[:, 0], outPos[:, 1], 64, range=[[-1, 1], [-1, 1]])
    return hist.T


def visualize_local_surface_information(sample_index, poly_order, coordinate_frame, scene_dir, traindata_file, use_tensorflow):
    polygons = vae.utils.load_polygons(scene_dir)
    with open(traindata_file + '.pickle', 'rb') as f:
        train_data = pickle.load(f)

    inPos = train_data['inPos'][sample_index]
    inNormal = train_data['inNormal'][sample_index]
    inDir = train_data['inDir'][sample_index]
    outPos = train_data['outPos'][sample_index]
    outDir = train_data['outDir'][sample_index]
    polyIdx = train_data['polygonIdx'][sample_index][0]

    tangent_space = coordinate_frame == 'tangent'

    if coordinate_frame == 'tangent':
        feat_name = 'mlsPoly{}Ts'.format(poly_order)
    elif coordinate_frame == 'light':
        feat_name = 'mlsPoly{}Ls'.format(poly_order)
    else:
        feat_name = 'mlsPoly{}'.format(poly_order)

    with open(traindata_file + '_{}.pickle'.format(feat_name), 'rb') as f:
        shapeCoeffs = pickle.load(f)

    sampled_p, sampled_n = vae.utils.sample_polygon(polygons[polyIdx], 5000)
    x, y = np.meshgrid(np.linspace(-1, 1, 64), np.linspace(-1, 1, 64))
    coords = np.stack([x.ravel(), y.ravel()], 1)
    _, coeffs = vae.utils.implicit_function_gradient_constraint(inPos, sampled_p, sampled_n, poly_order)
    f_taylor_ref = vae.utils.polynomial_to_voxel_grid(inPos, None, coeffs, poly_order, False)

    loaded_coeffs = shapeCoeffs[sample_index]

    # if tangent_space:
    #     loaded_coeffs = vae.utils.rotate_polynomial(loaded_coeffs, inNormal, 2, poly_order)

    if coordinate_frame == 'light':
        print('np.sum(inDir ** 2): {}'.format(np.sum(inDir ** 2)))
        inDir = inDir / np.sqrt(np.sum(inDir ** 2))
        f_taylor = vae.utils.polynomial_to_voxel_grid(inPos, -inDir, loaded_coeffs, poly_order, True)
    else:
        f_taylor = vae.utils.polynomial_to_voxel_grid(inPos, inNormal, loaded_coeffs, poly_order, tangent_space)

    f_taylor_rel = vae.utils.polynomial_to_voxel_grid(
        inPos * 0.0,  np.array([[1, 0]]), loaded_coeffs, poly_order, False)

    if use_tensorflow:
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            f_taylor_rel = vae.utils.tf_polynomial_to_voxel_grid(inPos[np.newaxis, :] * 0.0, np.array([[1, 0]]),
                                                                 loaded_coeffs[np.newaxis, :], poly_order, False)
            f_taylor_rel = sess.run(f_taylor_rel)
            f_taylor_rel = f_taylor_rel[0, :, :, 0]
    else:
        f_taylor_rel = vae.utils.polynomial_to_voxel_grid(
            inPos * 0.0,  np.array([[1, 0]]), loaded_coeffs, poly_order, False)

    fig, ax = plt.subplots(1, 2, figsize=(15, 15))
    ax[0].add_patch(Polygon(polygons[polyIdx], True, fill=False, color='red', alpha=0.5))
    ax[0].imshow(f_taylor, extent=[-1, 1, -1, 1], origin='bottom')
    ax[0].contour(f_taylor, levels=[0], extent=[-1, 1, -1, 1], colors='r')
    ax[0].scatter(inPos[0], inPos[1], color='r')
    ax[0].arrow(inPos[0], inPos[1], 0.2 * inDir[0], 0.2 * inDir[1], head_width=0.05, head_length=0.1, fc='k', ec='k')
    ax[0].set_title('Loaded Coeffs Polynomial')

    ax[1].add_patch(Polygon(polygons[polyIdx], True, fill=False, color='red', alpha=0.5))
    ax[1].imshow(f_taylor_ref, extent=[-1, 1, -1, 1], origin='bottom')
    ax[1].contour(f_taylor_ref, levels=[0], extent=[-1, 1, -1, 1], colors='r')
    ax[1].scatter(inPos[0], inPos[1], color='r')
    ax[1].set_title('Recomputed Coeffs Polynomial')

    fig, ax = plt.subplots(1, 3, figsize=(15, 15))
    ax[0].imshow(f_taylor_rel, extent=[-1, 1, -1, 1], origin='bottom')
    ax[0].contour(f_taylor_rel, levels=[0], extent=[-1, 1, -1, 1], colors='r')
    ax[0].scatter(0, 0, color='r')
    ax[0].set_title('Relative Polynomial')

    ax[1].imshow(f_taylor_rel <= 0, extent=[-1, 1, -1, 1], origin='bottom')
    ax[1].set_title('Discretized Relative Polynomial')

    discrete = vae.utils.discretize_polygon(polygons[polyIdx], 64)
    discrete = np.pad(discrete, [32, 32], 'constant')
    crd = ((inPos + 2) / 4 * 128).astype(np.int32)
    discrete = discrete[crd[1] - 32:crd[1] + 32, crd[0] - 32:crd[0] + 32]

    ax[2].imshow(discrete, extent=[-1, 1, -1, 1], origin='bottom')
    ax[2].set_title('Discretized Polygon')

    plt.show()


# Plot local implicit function taylor expansion
def visualize_local_geometry(scene_dir, poly_idx, path_t, poly_order):

    kernel = 'invdist2'
    epsilon = 0.01
    polygons = vae.utils.load_polygons(scene_dir)
    poly = polygons[poly_idx]

    sampled_p, sampled_n = vae.utils.sample_polygon(poly, 1000)

    current_pos, _, _ = vae.utils.index_polygon(path_t, poly)
    current_pos = np.atleast_2d(current_pos)
    plt.figure()

    # Evaluate taylor expansion
    x, y = np.meshgrid(np.linspace(-1, 1, 64), np.linspace(-1, 1, 64))
    coords = np.stack([x.ravel(), y.ravel()], 1)
    _, coeffs = vae.utils.implicit_function_gradient_constraint(current_pos, sampled_p, sampled_n,
                                                                poly_order, epsilon, kernel)

    f_taylor = vae.utils.polynomial_to_voxel_grid(current_pos, None, coeffs, poly_order, False)

    plt.imshow(f_taylor, extent=[-1, 1, -1, 1], origin='bottom')
    plt.colorbar()
    plt.contour(f_taylor, levels=[0], extent=[-1, 1, -1, 1], colors='orange')
    plt.scatter(current_pos[:, 0], current_pos[:, 1], color='r')
    plt.gca().add_patch(Polygon(poly, True, fill=False, color='red', alpha=0.9))


def visualize_reconstructed_samples(sess, path_t, poly_idx, config, trainer, vae_out_pos, scene_dir,
                                    traindata_file, dataset):

    feature_statistics = get_feature_stats(traindata_file)

    polygons = vae.utils.load_polygons(scene_dir)
    pos = polygons[poly_idx]
    with open(traindata_file + '.pickle', 'rb') as f:
        train_data = pickle.load(f)
    polyIdx = train_data['polygonIdx']

    # Load feature data
    if config.shape_features_name:
        with open(traindata_file + '_{}.pickle'.format(config.shape_features_name), 'rb') as f:
            shape_features = pickle.load(f)

    out_pos_mean = feature_statistics['outPosRel{}_mean'.format(config.prediction_space)]
    out_pos_stdinv = feature_statistics['outPosRel{}_stdinv'.format(config.prediction_space)]

    ref_pos, _, _ = vae.utils.index_polygon(path_t, pos)
    thresh = 0.00085

    indices = polyIdx[:, 0] == [poly_idx]

    for k in train_data:
        train_data[k] = train_data[k][indices, :]
    if config.shape_features_name:
        shape_features = shape_features[indices, :]

    # Select samples which are close to the current evaluation position
    indices = ((train_data['inPos'][:, 0] - ref_pos[0]) ** 2 < thresh) * \
        ((train_data['inPos'][:, 1] - ref_pos[1]) ** 2 < thresh)
    for k in train_data:
        train_data[k] = train_data[k][indices, :]
    if config.shape_features_name:
        shape_features = shape_features[indices, :]

    samples = []
    feed_dict = {}
    batch_size = 128
    feed_dict[trainer.in_pos_p] = train_data['inPos'][:batch_size, :]
    feed_dict[trainer.in_dir_p] = train_data['inDir'][:batch_size, :]
    feed_dict[trainer.phase_p] = False
    feed_dict[trainer.in_normal_p] = train_data['inNormal'][:batch_size, :]
    feed_dict[trainer.out_pos_p] = train_data['outPos'][:batch_size, :]
    feed_dict[trainer.polygon_idx] = train_data['polygonIdx'][:batch_size, :]
    feed_dict[trainer.out_pos_mean_p] = out_pos_mean
    feed_dict[trainer.out_pos_stdinv_p] = out_pos_stdinv
    # feed_dict[trainer.albedo_p] = train_data['albedo'][:batch_size, :]
    feed_dict[trainer.albedo_p] = np.ones((batch_size, 3)) * 0.6
    feed_dict[trainer.albedo_mean_p] = feature_statistics['albedo_mean']
    feed_dict[trainer.albedo_stdinv_p] = feature_statistics['albedo_stdinv']
    # feed_dict[trainer.sigma_t_p] = train_data['sigma_t'][:batch_size, :]
    feed_dict[trainer.sigma_t_p] = np.ones((batch_size, 3)) * 10
    feed_dict[trainer.sigma_t_mean_p] = feature_statistics['sigmaT_mean']
    feed_dict[trainer.sigma_t_stdinv_p] = feature_statistics['sigmaT_mean']

    if config.shape_features_name:
        feed_dict[trainer.shape_features_p] = shape_features[:batch_size, :]
        f = config.shape_features_name
        feed_dict[trainer.shape_features_mean_p] = feature_statistics['{}_mean'.format(f)]
        feed_dict[trainer.shape_features_stdinv_p] = feature_statistics['{}_stdinv'.format(f)]

    reconstructed_samples = sess.run(
        vae_out_pos, feed_dict=feed_dict)
    samples.append(reconstructed_samples)

    samples = np.array(samples)[0, :, :]

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.add_patch(Polygon(pos, True, fill=False, color='red', alpha=0.2))
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.gca().set_aspect('equal', adjustable='box')

    hist, _, _ = np.histogram2d(samples[:, 0], samples[:, 1], 64, range=[[-1, 1], [-1, 1]])
    hist = np.flipud(hist.T)
    plt.imshow(hist, extent=[-1, 1, -1, 1])
    plt.title('Reconstructed Samples')

def visualize_reconstructed_samples_3d(sess, mesh_name, config, trainer, model_outputs, dataset):
    # Generate some new training samples from the given mesh
    
    train_data = dataset.sample(mesh_name, 0, 128)
    feature_statistics = dataset.get_feature_statistics()
    feed_dict = {}
    batch_size = 128
    feed_dict[trainer.in_pos_p] = train_data['inPos'][:batch_size, :]
    feed_dict[trainer.out_pos_p] = train_data['outPos'][:batch_size, :]
    feed_dict[trainer.polygon_idx] = train_data['meshIdx'][:batch_size, :]
    feed_dict[trainer.in_normal_p] = train_data['inNormal'][:batch_size, :]
    feed_dict[trainer.in_dir_p] = train_data['inDir'][:batch_size, :]
    feed_dict[trainer.phase_p] = False

    if config.shape_features_name:
        feed_dict[trainer.shape_features_p] = train_data[config.shape_features_name][:batch_size, :]
        f = config.shape_features_name
        feed_dict[trainer.shape_features_mean_p] = feature_statistics['{}_mean'.format(f)]
        feed_dict[trainer.shape_features_stdinv_p] = feature_statistics['{}_stdinv'.format(f)]

    feed_dict[trainer.out_pos_mean_p] = feature_statistics['outPosRel{}_mean'.format(config.prediction_space)]
    feed_dict[trainer.out_pos_stdinv_p] = feature_statistics['outPosRel{}_stdinv'.format(config.prediction_space)]

    feed_dict[trainer.albedo_p] = train_data['albedo'][:batch_size, :]
    feed_dict[trainer.albedo_mean_p] = feature_statistics['albedo_mean']
    feed_dict[trainer.albedo_stdinv_p] = feature_statistics['albedo_stdinv']
    feed_dict[trainer.eff_albedo_p] = train_data['effAlbedo'][:batch_size, :]
    feed_dict[trainer.eff_albedo_mean_p] = feature_statistics['effAlbedo_mean']
    feed_dict[trainer.eff_albedo_stdinv_p] = feature_statistics['effAlbedo_stdinv']
    feed_dict[trainer.sigma_t_p] = train_data['sigmaT'][:batch_size, :]
    feed_dict[trainer.sigma_t_mean_p] = feature_statistics['sigmaT_mean']
    feed_dict[trainer.sigma_t_stdinv_p] = feature_statistics['sigmaT_mean']
    feed_dict[trainer.g_p] = train_data['g'][:batch_size, :]
    feed_dict[trainer.g_mean_p] = feature_statistics['g_mean']
    feed_dict[trainer.g_stdinv_p] = feature_statistics['g_stdinv']
    feed_dict[trainer.ior_p] = train_data['ior'][:batch_size, :]

    samples = []
    reconstructed_samples = sess.run(
        model_outputs['out_pos'], feed_dict=feed_dict)
    samples.append(reconstructed_samples)
    samples = np.array(samples)[0, :, :]
    return samples, train_data['outPos'][:batch_size, :], train_data[config.shape_features_name][:batch_size, :], train_data['inDir'][:batch_size, :]
