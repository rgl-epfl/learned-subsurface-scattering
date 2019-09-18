#!/usr/bin/env python3

import glob
import json
import os
import pickle
import random
import subprocess
import sys
import copy
from shutil import copyfile

import numpy as np
import tensorflow as tf
import tqdm
import trimesh
from matplotlib.patches import Polygon

import mitsuba
import vae.config
import vae.trainer
from mitsuba.core import *
from utils.experiments import dump_config, get_existing_net, load_config
from utils.mesh import *
from utils.tensorflow import bytes_feature
from vae.model import generate_new_samples
from vae.utils import (discretize_polygon,
                       implicit_function_gradient_constraint,
                       implicit_function_iterated,
                       implicit_function_polynomial, load_polygons, project_points_on_poly,
                       rescale_polygon, sample_polygon,
                       shape_feat_name_to_num_coeff, world_to_local_np, extract_poly_order_from_feat_name)
import vae.utils
from vae.global_config import *
import utils.math
import vae.datapipeline
from vae.utils import project_points_on_mesh
from utils.math import onb_duff
import vae.tf_utils

from utils.printing import printr


try:
    import matplotlib.pyplot as plt
except Exception as e:
    # See: https://stackoverflow.com/questions/4931376/generating-matplotlib-graphs-without-a-running-x-server/4935945#4935945
    import matplotlib
    matplotlib.use('Agg', force=True, warn=False)
    import matplotlib.pyplot as plt


def shuffle_dict_of_arrays(data, batch_size, uniform_batches):
    indices = np.arange(0, data[next(iter(data))].shape[0], 1)
    if uniform_batches:
        indices = np.reshape(indices, [-1, batch_size])
        np.random.shuffle(indices)
        indices = indices.ravel()
    else:
        np.random.shuffle(indices)

    for k in data:
        data[k] = data[k][indices, :]


def random_albedo():
    """Randomly samples albedo by sampling uniformly in effective albedo space and remapping
      to single scattering albedo (PBRT v3 Eq. 15.48 p. 935)"""
    effective_albedo = np.random.rand()
    single_scatter_albedo = (1 - np.exp(-8 * effective_albedo)) / (1 - np.exp(-8))
    return single_scatter_albedo


def sample_scattering_2d(polygons, n_train_samples, uniform_batches, batch_size, vary_volume_params):
    tmp_result = mitsuba.render.Volpath2D.sample([mitsuba.core.Point2(r[0], r[1]) for r in polygons[0]], [
                                                 mitsuba.core.Spectrum(1.0)], [mitsuba.core.Spectrum(15.0)], 1, True, 1)
    train_data = {k: [] for k in tmp_result.keys()}
    spectrum_features = [k for k in tmp_result if type(tmp_result[k][0]) == mitsuba.core.Spectrum]
    vector_features = [k for k in tmp_result if type(
        tmp_result[k][0]) == mitsuba.core.Vector2 or type(tmp_result[k][0]) == mitsuba.core.Point2]
    other_features = [k for k in tmp_result if k not in spectrum_features and k not in vector_features]
    train_data['polygonIdx'] = []

    for poly_idx, polygon in tqdm.tqdm(enumerate(polygons)):
        if vary_volume_params:
            current_albedo = [mitsuba.core.Spectrum(random_albedo())]
            current_sigma_t = [mitsuba.core.Spectrum(10.0)]
        else:
            current_albedo = [mitsuba.core.Spectrum(1.0)]
            current_sigma_t = [mitsuba.core.Spectrum(15.0)]

        result = mitsuba.render.Volpath2D.sample([mitsuba.core.Point2(
            r[0], r[1]) for r in polygon], current_albedo, current_sigma_t, n_train_samples, uniform_batches, batch_size)

        for i in range(len(result['inPos'])):
            for k in spectrum_features:
                train_data[k].append(np.array([result[k][i][0], result[k][i][1], result[k][i][2]]))
            for k in vector_features:
                train_data[k].append(np.array([result[k][i].x, result[k][i].y]))
            for k in other_features:
                train_data[k].append(result[k][i])
            train_data['polygonIdx'].append(poly_idx)
    for k in train_data:
        train_data[k] = np.array(train_data[k])
        if train_data[k].ndim == 1:
            train_data[k] = train_data[k][:, np.newaxis]

    train_data['bounces'] = train_data['bounces'].astype(np.int32)
    train_data['polygonIdx'] = train_data['polygonIdx'].astype(np.int32)
    n_usable = train_data['inPos'].shape[0] // batch_size * batch_size
    print('max train_data[absorptionProbVar]: {}'.format(np.max(train_data['absorptionProbVar'])))
    train_data.pop('absorptionProbVar', None)

    for k in train_data:
        train_data[k] = train_data[k][:n_usable, :]
    return train_data, n_usable


def extract_shape_features_2d(train_data, polygons, n_usable, batch_size, used_poly_orders):
    """ Generate polynom fits of different orders """

    # For each training sample, extract the coefficients of a locally fit polynomial
    surface_points, surface_normals = [], []
    for poly_idx, polygon in tqdm.tqdm(enumerate(polygons)):
        sampled_p, sampled_n = sample_polygon(polygon, 15000)
        surface_points.append(sampled_p)
        surface_normals.append(sampled_n)

    def mls(inPos, surface_points, surface_normals, poly_order):
        _, coeffs = implicit_function_gradient_constraint(
            inPos, surface_points, surface_normals, poly_order, sigma2=0.1, kernel='gaussian', reg=1e-4)
        return coeffs.T

    shape_feat_extractors = []
    shape_feat_extractors += [(deg, 'mlsPoly', lambda ip, pp, pn,
                               deg=deg: mls(ip, pp, pn, deg)) for deg in used_poly_orders]

    shape_features_all = {}
    for poly_order, feat_name, feat_extractor in shape_feat_extractors:
        shape_features, shape_features_ts, shape_features_ls = [], [], []
        for i in tqdm.tqdm(range(int(n_usable // batch_size))):
            p = train_data['polygonIdx'][i * batch_size, 0]
            # Transform all data to local frame
            coeffs = feat_extractor(train_data['inPos'][i * batch_size],
                                    surface_points[p], surface_normals[p])

            # tangent space and light space surface description
            coeffs_ts = vae.utils.rotate_polynomial(coeffs, train_data['inNormal'][i * batch_size], 2, poly_order)
            coeffs_light = vae.utils.rotate_polynomial(coeffs, -train_data['inDir'][i * batch_size], 2, poly_order)
            coeffs = np.repeat(coeffs, batch_size, 0)  # Same coeffs for the whole batch
            coeffs_ts = np.repeat(coeffs_ts, batch_size, 0)
            coeffs_light = np.repeat(coeffs_light, batch_size, 0)
            shape_features.append(coeffs)
            shape_features_ts.append(coeffs_ts)
            shape_features_ls.append(coeffs_light)
        shape_features_all['{}{}'.format(feat_name, poly_order)] = np.concatenate(shape_features, 0)
        shape_features_all['{}TS{}'.format(feat_name, poly_order)] = np.concatenate(shape_features_ts, 0)
        shape_features_all['{}LS{}'.format(feat_name, poly_order)] = np.concatenate(shape_features_ls, 0)

    return shape_features_all


def get_stats_from_dict(data, constant_variables=[]):

    def zero_variance(var, feat_name):
        smoothing_epsilon = np.ones_like(var) * 1e-1
        var[var == 0.0] = 1.0
        if feat_name in constant_variables:
            var = np.ones_like(var)
            smoothing_epsilon = 0.0

        if feat_name + 'X' in constant_variables:
            var[0] = 1.0
            smoothing_epsilon[0] = 0.0
        if feat_name + 'Y' in constant_variables:
            var[1] = 1.0
            smoothing_epsilon[1] = 0.0
        if feat_name + 'Z' in constant_variables:
            var[2] = 1.0
            smoothing_epsilon[2] = 0.0
        return var, 1.0 / np.sqrt(var + smoothing_epsilon)

    def set_mean_var(stats_dict, name, values):
        var = np.var(values, 0)
        var, invstd = zero_variance(var, name)
        stats_dict[f'{name}_mean'] = np.mean(values, 0)
        stats_dict[f'{name}_stdinv'] = invstd

    stats_dict = {}
    n_samples = data['inPos'].shape[0]

    for k in data:
        if k in ['points', 'pointNormals', 'pointWeights']:
            mean = np.mean(data[k], (0, 1))
            var = np.var(data[k], (0, 1))
            if mean.size == 1:
                mean = mean[None]
                var = var[None]
        else:
            mean = np.mean(data[k], 0)
            var = np.var(data[k], 0)
        stats_dict['{}_mean'.format(k)] = mean
        var, invstd = zero_variance(var, k)
        stats_dict['{}_stdinv'.format(k)] = invstd

    local_out_pos = vae.utils.world_to_local_np_new(data['outPos'], data['inPos'], data['inDir'], data['inNormal'], 'WS')
    local_out_pos_ts = vae.utils.world_to_local_np_new(data['outPos'], data['inPos'], data['inDir'], data['inNormal'], 'TS')
    local_out_pos_ls = vae.utils.world_to_local_np_new(data['outPos'], data['inPos'], data['inDir'], data['inNormal'], 'LS')
    # TODO: Currently not supported
    # local_out_pos_as = vae.utils.world_to_local_np_new(data['outPos'], data['inPos'], data['inDir'], data['inNormal'], 'AS')

    var = np.var(local_out_pos, 0)
    var, invstd = zero_variance(var, 'outPosRelWS')
    stats_dict['outPosRelWS_mean'] = np.mean(local_out_pos, 0)
    stats_dict['outPosRelWS_stdinv'] = invstd
    var = np.var(local_out_pos_ts, 0)
    var, invstd = zero_variance(var, 'outPosRelTS')
    stats_dict['outPosRelTS_mean'] = np.mean(local_out_pos_ts, 0)
    stats_dict['outPosRelTS_stdinv'] = invstd
    var = np.var(local_out_pos_ls, 0)
    var, invstd = zero_variance(var, 'outPosRelLS')
    stats_dict['outPosRelLS_mean'] = np.mean(local_out_pos_ls, 0)
    stats_dict['outPosRelLS_stdinv'] = invstd
    # var = np.var(local_out_pos_as, 0)
    # var, invstd = zero_variance(var, 'outPosRelAS')
    # stats_dict['outPosRelAS_mean'] = np.mean(local_out_pos_as, 0)
    # stats_dict['outPosRelAS_stdinv'] = invstd

    if 'points' in data:
        points_rel = world_to_local_np(data['inPos'][:, None, :], data['inNormal'][:, None, :], data['points'], False)
        points_ts = world_to_local_np(data['inPos'][:, None, :], data['inNormal'][:, None, :], data['points'], True)
        points_ls = world_to_local_np(data['inPos'][:, None, :], data['inDir'][:, None, :], data['points'], True)
        set_mean_var(stats_dict, 'pointsWS', np.reshape(points_rel, [-1, 3]))
        set_mean_var(stats_dict, 'pointsTS', np.reshape(points_ts, [-1, 3]))
        set_mean_var(stats_dict, 'pointsLS', np.reshape(points_ls, [-1, 3]))
        normals_ts = world_to_local_np(data['inPos'][:, None, :] * 0.0, data['inNormal']
                                       [:, None, :], data['pointNormals'], True)
        normals_ls = world_to_local_np(data['inPos'][:, None, :] * 0.0, data['inDir']
                                       [:, None, :], data['pointNormals'], True)
        set_mean_var(stats_dict, 'pointNormalsTS', np.reshape(normals_ts, [-1, 3]))
        set_mean_var(stats_dict, 'pointNormalsLS', np.reshape(normals_ls, [-1, 3]))

    return stats_dict, n_samples


def generate_training_data(scene_dir, datasetname, output_dir, dataset_dir, process_only=False,
                           n_threads=12, cluster_job=False, job_id=0, n_jobs=1, generate_meshes_only=False):
    """Simulates scattering for all training polygons"""
    os.makedirs(scene_dir, exist_ok=True)
    scatter_data = vae.datapipeline.get_config(datasetname)
    scatter_data = scatter_data(scene_dir, RESOURCEDIR, dataset_dir)
    if generate_meshes_only:
        scatter_data.generate_meshes()
        return
    if not process_only:
        scatter_data.generate(n_threads, cluster_job, job_id, n_jobs)
    if not cluster_job:  # Do not run the processing step on the cluster before all jobs are finished
        scatter_data.process()


def feat_infos(config):
    info = [('inPos', tf.float32, config.dim),
            ('inNormal', tf.float32, config.dim),
            ('inDir', tf.float32, config.dim),
            ('outPos', tf.float32, config.dim),
            ('throughput', tf.float32, 3),
            ('albedo', tf.float32, 3),
            ('effAlbedo', tf.float32, 1),
            ('sigmaT', tf.float32, 3),
            ('g', tf.float32, 1),
            ('ior', tf.float32, 1),
            ('absorptionProb', tf.float32, 1),
            ('outDir', tf.float32, config.dim),
            ('outNormal', tf.float32, config.dim),
            ('polygonIdx', tf.int32, 1) if config.dim == 2 else ('meshIdx', tf.int32, 1)]
    if config.shape_features_name:
        info.append((config.shape_features_name, tf.float32,
                     shape_feat_name_to_num_coeff(config.shape_features_name, config.dim)))

    return info


def batch_indices(config):
    info = feat_infos(config)
    return {info[i][0]: i for i in range(len(info))}


def decode_tfrecord(data, config, absorption_only):
    """Helper function to decode a stored tf record"""
    feature_infos = feat_infos(config)
    feature = {f[0]: tf.FixedLenFeature([], tf.string) for f in feature_infos}
    features = tf.parse_single_example(data, features=feature)
    out_features = []
    for f in feature_infos:
        feat = tf.decode_raw(features[f[0]], f[1])

        if len(f) == 4:
            feat = tf.reshape(feat, [-1, f[2], f[3]])
        else:
            feat = tf.reshape(feat, [-1, f[2]])

        if absorption_only:
            out_features.append(feat[0, :])
        else:
            out_features.append(feat)
    return tuple(out_features)


def get_dataset_iterator(train_data_file, batch_size, numepochs, config, feature_statistics, absorption_only, test=False):
    """Produces an iterator to iterate over the training data"""
    # read training data from disk
    dataset = tf.data.TFRecordDataset(train_data_file)

    def _decode(data):
        return decode_tfrecord(data, config, absorption_only)

    dataset = dataset.map(_decode)

    if not absorption_only:
        dataset = dataset.apply(tf.contrib.data.unbatch())

    shuffle_buffer_size = 4096

    if not config.use_wae_mmd:
        dataset = dataset.shuffle(shuffle_buffer_size)

    # If requested, filter out outlier positions

    if config.filter_outliers:
        idx = batch_indices(config)
        # 4 standard deviations: Only remove worst outliers
        cutoff = config.outlier_distance / feature_statistics['outPosRelWS_stdinv']

        def filter_outliers(*args):
            outPos = args[idx['outPos']]
            inPos = args[idx['inPos']]
            return tf.reduce_all(tf.abs(outPos - inPos) < cutoff)
        dataset = dataset.filter(filter_outliers)

    # Filter out datasamples if they have outlier features
    # At test time, do not exclude any data samples
    if config.filter_feature_outliers and not test and config.shape_features_name is not None:
        idx = batch_indices(config)
        cutoff_coeffs = config.outlier_distance / feature_statistics[config.shape_features_name + '_stdinv']

        def filter_outliers(*args):
            shape_feats = args[idx[config.shape_features_name]]
            delta_feats = tf.abs(shape_feats - feature_statistics[config.shape_features_name + '_mean'])
            return tf.reduce_all(delta_feats < cutoff_coeffs)
        dataset = dataset.filter(filter_outliers)

    dataset = dataset.batch(batch_size)
    if not absorption_only:
        dataset = dataset.shuffle(shuffle_buffer_size)
    dataset = dataset.repeat(numepochs)

    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()
    return next_element


def draw_polygons(scene_dir):
    x = np.arange(-10, 10)
    y = x ** 2

    fig = plt.figure()
    ax = fig.add_subplot(111)

    coords = []

    plt.xlim(-1, 1)
    plt.ylim(-1, 1)

    os.makedirs(scene_dir, exist_ok=True)

    def onclick(event):
        ix, iy = event.xdata, event.ydata
        print('event.button: {}'.format(event.button))
        if event.button == 1:
            print('ix: {}'.format(ix))
            print('iy: {}'.format(iy))
            coords.append((ix, iy))
            coords_np = np.array(coords)

        else:
            coords_np = np.array(coords)
            idx = np.argmin((coords_np[:, 0] - ix) ** 2 + (coords_np[:, 1] - iy) ** 2)
            print('idx: {}'.format(idx))
            del coords[idx]
            coords_np = np.array(coords)

        ax.clear()
        plt.xlim(-1, 1)
        plt.ylim(-1, 1)
        if len(coords) > 0:
            ax.add_patch(Polygon(coords, True))
            plt.scatter(coords_np[:, 0], coords_np[:, 1])
        plt.draw()
        return coords

    def onkey(event):
        print('event: {}'.format(event.key))
        if event.key == 'r':
            coords.clear()
            ax.clear()
            plt.draw()

        if event.key == 'a':
            # Write out the polygon to file
            existing_last = sorted(glob.glob(scene_dir + '/polygon*.pickle'))
            if len(existing_last) > 0:
                last_idx = int(existing_last[-1][-10:-7])
                new_idx = last_idx + 1
            else:
                new_idx = 1
            with open(os.path.join(scene_dir, 'polygon{:03}.pickle'.format(new_idx)), 'wb') as f:
                pickle.dump(np.array(coords), f)

    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    cid = fig.canvas.mpl_connect('key_press_event', onkey)
    plt.show()


def fix_polygon_winding(tmp_dir):

    fix_polygon_winding.poly_files = sorted(glob.glob(tmp_dir + '/scenes/poly*.pickle'))

    def load_poly(idx):
        with open(fix_polygon_winding.poly_files[idx], 'rb') as f:
            return rescale_polygon(pickle.load(f))

    def save_poly(idx, poly):
        with open(fix_polygon_winding.poly_files[idx], 'wb') as f:
            pickle.dump(poly, f)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    fix_polygon_winding.poly_idx = 0
    poly = load_poly(fix_polygon_winding.poly_idx)
    ax.clear()
    ax.add_patch(Polygon(poly, True))
    ax.scatter(poly[:, 0], poly[:, 1])
    for i, txt in enumerate(np.arange(0, poly.shape[0], 1)):
        ax.annotate(txt, (poly[i, 0], poly[i, 1]))
    plt.draw()

    def onkey(event):
        if event.key == 'a':
            print('All okay')

        elif event.key == 'd':
            print('Fix winding')
            save_poly(fix_polygon_winding.poly_idx, np.flipud(load_poly(fix_polygon_winding.poly_idx)))

        fix_polygon_winding.poly_idx += 1
        if fix_polygon_winding.poly_idx >= len(fix_polygon_winding.poly_files):
            return

        poly = load_poly(fix_polygon_winding.poly_idx)
        ax.clear()
        ax.add_patch(Polygon(poly, True))
        ax.scatter(poly[:, 0], poly[:, 1])
        for i, txt in enumerate(np.arange(0, poly.shape[0], 1)):
            ax.annotate(txt, (poly[i, 0], poly[i, 1]))
        plt.draw()

    cid = fig.canvas.mpl_connect('key_press_event', onkey)
    plt.show()
