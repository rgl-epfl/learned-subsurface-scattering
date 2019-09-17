import glob
import json
import os
import pickle
import random
import multiprocessing
import time

import numpy as np
import tensorflow as tf
import tqdm
import trimesh

import mitsuba
import utils.math
import vae.datahandler
import vae.datapipeline
import vae.utils
from mitsuba.core import *
from utils.math import onb_duff
from utils.mesh import sample_mesh
from utils.tensorflow import bytes_feature
from vae.global_config import *
from vae.utils import (discretize_polygon, extract_poly_order_from_feat_name,
                       implicit_function_gradient_constraint,
                       implicit_function_iterated,
                       implicit_function_polynomial, load_polygons, project_points_on_mesh,
                       project_points_on_poly, rescale_polygon, sample_polygon,
                       shape_feat_name_to_num_coeff)

import utils.mtswrapper


class MediumParamGenerator:
    def get_g(self):
        return float(np.random.rand(1)) * MAX_G

    def get_albedo(self):
        return mitsuba.core.Spectrum(np.maximum(vae.datahandler.random_albedo(), 0.05))

    def get_eta(self):
        return float(np.random.rand(1) * 0.5 + 1.0)

    def get_sigmat(self):
        return mitsuba.core.Spectrum(1.0)

    def get_medium(self):
        medium = utils.mtswrapper.MediumParameters()
        medium.albedo = self.get_albedo()
        medium.g = self.get_g()
        medium.ior = self.get_eta()
        medium.sigmaT = self.get_sigmat()
        return medium

    def get_media(self, nParams):
        result = []
        for i in range(nParams):
            medium = self.get_medium()
            result.append(medium)
        return result


class FixedMediumGenerator(MediumParamGenerator):
    def get_g(self):
        return 0.5

    def get_albedo(self):
        return mitsuba.core.Spectrum(0.99)

    def get_eta(self):
        return 1.0

    def get_sigmat(self):
        return mitsuba.core.Spectrum(1.0)


class SimilarityMediumParamGenerator(MediumParamGenerator):
    def reduce_params_to_orig(self, alpha_p, g, sigma_t_p):
        sigma_sp = alpha_p * sigma_t_p
        sigma_s = sigma_sp / (1 - g)
        sigma_a = sigma_t_p - sigma_sp
        sigma_t = sigma_s + sigma_a
        alpha = sigma_s / sigma_t
        return alpha, g, sigma_t

    def get_medium(self):
        alpha_p = self.get_albedo()[0]
        g = self.get_g()
        sigma_tp = self.get_sigmat()[0]
        albedo, g, sigmat = self.reduce_params_to_orig(alpha_p, g, sigma_tp)
        albedo = mitsuba.core.Spectrum(float(albedo))
        sigmat = mitsuba.core.Spectrum(float(sigmat))
        medium = utils.mtswrapper.MediumParameters()
        medium.albedo = albedo
        medium.g = float(g)
        medium.ior = self.get_eta()
        medium.sigmaT = sigmat
        return medium


class MixedMediumParamGenerator(SimilarityMediumParamGenerator):
    def get_medium(self):
        alpha_p = self.get_albedo()[0]
        g = self.get_g()
        sigma_tp = self.get_sigmat()[0]
        albedo, g, sigmat = self.reduce_params_to_orig(alpha_p, g, sigma_tp)
        albedo = mitsuba.core.Spectrum(float(albedo))
        sigmat = mitsuba.core.Spectrum(float(sigmat))
        medium = utils.mtswrapper.MediumParameters()
        medium.albedo = albedo
        medium.g = float(g)
        medium.ior = self.get_eta()
        medium.sigmaT = sigmat if np.random.rand(1) > 0.5 else mitsuba.core.Spectrum(1.0)
        return medium

class MixedMediumParamGenerator2(SimilarityMediumParamGenerator):
    def get_medium(self):
        alpha_p = self.get_albedo()[0]
        g = self.get_g()
        sigma_tp = self.get_sigmat()[0]
        albedo, g, sigmat = self.reduce_params_to_orig(alpha_p, g, sigma_tp)
        albedo = mitsuba.core.Spectrum(float(albedo))
        sigmat = mitsuba.core.Spectrum(float(sigmat))
        medium = utils.mtswrapper.MediumParameters()
        medium.albedo = albedo
        medium.g = float(g)
        medium.ior = self.get_eta()
        medium.sigmaT = sigmat if np.random.rand(1) > 0.8 else mitsuba.core.Spectrum(1.0)
        return medium


class MixedMediumParamGenerator3(SimilarityMediumParamGenerator):
    def get_medium(self):
        alpha_p = self.get_albedo()[0]
        g = self.get_g()
        sigma_tp = self.get_sigmat()[0]
        albedo, g, sigmat = self.reduce_params_to_orig(alpha_p, g, sigma_tp)
        albedo = mitsuba.core.Spectrum(float(albedo))
        sigmat = mitsuba.core.Spectrum(float(sigmat))
        medium = utils.mtswrapper.MediumParameters()
        medium.albedo = albedo
        medium.g = float(g)
        medium.ior = self.get_eta()
        medium.sigmaT = sigmat if np.random.rand(1) > 0.5 else mitsuba.core.Spectrum(1.0)
        medium.g =  medium.g if np.random.rand(1) > 0.5 else float(self.get_albedo()[0]* 0.95) 
        return medium

class MixedMediumParamGenerator4(SimilarityMediumParamGenerator):
    def get_medium(self):
        alpha_p = self.get_albedo()[0]
        g = self.get_g()
        sigma_tp = self.get_sigmat()[0]
        albedo, g, sigmat = self.reduce_params_to_orig(alpha_p, g, sigma_tp)
        albedo = mitsuba.core.Spectrum(float(albedo))
        sigmat = mitsuba.core.Spectrum(float(sigmat))
        medium = utils.mtswrapper.MediumParameters()
        medium.albedo = albedo
        medium.g = float(g)
        medium.ior = self.get_eta()
        medium.sigmaT = sigmat if np.random.rand(1) > 0.5 else mitsuba.core.Spectrum(1.0)
        medium.g =  medium.g if np.random.rand(1) > 0.75 else float(self.get_albedo()[0]* 0.95) 
        return medium


def save_sharded(mesh_idx, train_data, shard_size, traindata_file):
    n_samples = train_data['inPos'].shape[0]
    n_shards = n_samples // shard_size
    if n_shards * shard_size < n_samples:
        n_shards += 1
    os.makedirs(traindata_file, exist_ok=True)
    for i in range(n_shards):
        train_data_shard = {}
        for k in train_data.keys():
            train_data_shard[k] = train_data[k][(
                i * shard_size):np.minimum(((i + 1) * shard_size), n_samples), :]
        fn = traindata_file + '/data_{:06d}_{:06d}.pickle'.format(mesh_idx, i)
        with open(fn, 'wb') as f:
            pickle.dump(train_data_shard, f)


def mts_output_to_dict(mts_output, rename_dict={}):
    result = {}
    for k in mts_output:
        k_out = rename_dict[k] if k in rename_dict else k
        if type(mts_output[k][0]) in [Point, Vector, Spectrum]:
            result[k_out] = vae.utils.mts_to_np(mts_output[k])
        else:
            result[k_out] = np.array(mts_output[k])
        if result[k_out].ndim == 1:
            result[k_out] = result[k_out][:, np.newaxis]
        if result[k_out].dtype == np.float64:
            result[k_out] = result[k_out].astype(np.float32)
        elif result[k_out].dtype == np.int64:
            result[k_out] = result[k_out].astype(np.int32)
    return result


def build_constraint_kdtree(mesh_file, sigma_t=1.0, g=0.0, albedo=0.5, resource_dir='./resources'):
    if mesh_file is None:
        return None, None, None

    inv_kernel_eps = 1.0 / vae.utils.kernel_epsilon(g, sigma_t, albedo)
    sampled_p, sampled_n = sample_mesh(mesh_file, None, density=POINTDENSITY *
                                       inv_kernel_eps, resource_dir=resource_dir)
    sampled_p_mts = [vae.utils.mts_p(p) for p in sampled_p]
    sampled_n_mts = [vae.utils.mts_v(p) for p in sampled_n]
    constraint_kdtree = mitsuba.render.ConstraintKdTree()
    constraint_kdtree.build(sampled_p_mts, sampled_n_mts)
    return constraint_kdtree, sampled_p, sampled_n


def get_local_points(in_pos, in_pos_mts, batch_size, n_points, g, albedo, constraint_kdtree):

    effective_epsilon = vae.utils.kernel_epsilon(g, 1, albedo)
    all_pts, all_nors, _ = mitsuba.render.Volpath3D.getLocalPoints(
        in_pos_mts, effective_epsilon, 'gaussian', constraint_kdtree)

    all_pts = vae.utils.mts_to_np(all_pts)
    all_nors = vae.utils.mts_to_np(all_nors)
    all_weights = np.exp(-np.sum((in_pos - all_pts) ** 2, axis=1) / (2 * effective_epsilon))
    ret_pts = np.zeros((batch_size, n_points, 3))
    ret_nors = np.zeros((batch_size, n_points, 3))
    ret_weights = np.zeros((batch_size, n_points))
    for b in range(batch_size):
        pt_indices = utils.math.weighted_sampling_without_replacement(all_weights.tolist(), n_points)
        pts = all_pts[pt_indices][None, :, :]
        nors = all_nors[pt_indices][None, :, :]
        weights = all_weights[pt_indices][None, :]
        if pts.shape[0] < n_points:  # Pad the arrays to n_points
            pts = np.concatenate([pts, np.zeros((1, n_points - pts.shape[1], 3))], 1)
            nors = np.concatenate([nors, np.zeros((1, n_points - nors.shape[1], 3))], 1)
            weights = np.concatenate([weights, np.zeros((1, n_points - weights.shape[1]))], 1)
        ret_pts[b, :, :] = pts
        ret_nors[b, :, :] = nors
        ret_weights[b, :] = weights
    return ret_pts, ret_nors, ret_weights


def sample_scattering_poly_3d(mesh_file, mesh_idx, n_train_samples, batch_size, n_abs_samples,
                              medium_param_generator, extra_options):

    poly_order = extra_options['order'] if 'order' in extra_options else 3
    fixed_polynomial = extra_options['fixed_polynomial'] if 'fixed_polynomial' in extra_options else None
    seed = extra_options['seed'] if 'seed' in extra_options else 123
    gather_point_cloud = extra_options['gather_point_cloud'] if 'gather_point_cloud' in extra_options else False

    constraint_kdtree, _, _ = build_constraint_kdtree(mesh_file)

    nParams = n_train_samples // batch_size

    media = medium_param_generator.get_media(nParams)

    if fixed_polynomial is not None:
        result = utils.mtswrapper.sample_scattering_poly(
            n_train_samples, batch_size, n_abs_samples, media, fixed_polynomial, seed, extra_options)
    else:
        result = utils.mtswrapper.sample_scattering_mesh(
            mesh_file, n_train_samples, batch_size, n_abs_samples, media, seed, constraint_kdtree, extra_options)

    n_gen_samples = len(result['inPos'])
    print('Done sampling')
    print('NSamples: {}'.format(n_gen_samples))
    train_data = mts_output_to_dict(result, {'shapeCoeffs': f'mlsPoly{poly_order}'})
    train_data['meshIdx'] = np.ones((n_gen_samples, 1), dtype=np.int32) * mesh_idx
    train_data['effAlbedo'] = vae.utils.albedo_to_effective_albedo(
        train_data['albedo'])[..., 0][..., np.newaxis]
    train_data['albedop'] = vae.utils.get_alphap(
        train_data['albedo'], train_data['g'], train_data['sigmaT'])[..., 0][..., np.newaxis]

    train_data.pop('absorptionProbVar', None)
    # Go over all the training samples and extract light space polynomial
    train_data[f'mlsPolyLS{poly_order}'] = np.zeros_like(
        train_data[f'mlsPoly{poly_order}'])
    train_data[f'mlsPolyTS{poly_order}'] = np.zeros_like(
        train_data[f'mlsPoly{poly_order}'])
    train_data[f'mlsPolyAS{poly_order}'] = np.zeros_like(
        train_data[f'mlsPoly{poly_order}'])
    n_points = 64
    if gather_point_cloud:
        train_data['points'] = np.zeros((n_gen_samples, n_points, 3), dtype=np.float32)
        train_data['pointNormals'] = np.zeros((n_gen_samples, n_points, 3), dtype=np.float32)
        train_data['pointWeights'] = np.zeros((n_gen_samples, n_points), dtype=np.float32)

    for i in range(int(train_data['inPos'].shape[0] // batch_size)):
        coeffs_ws = np.array(train_data[f'mlsPoly{poly_order}'][i * batch_size])
        in_dir = train_data['inDir'][i * batch_size]
        in_normal = train_data['inNormal'][i * batch_size]

        coeffs = utils.mtswrapper.rotate_polynomial(coeffs_ws, -in_dir, poly_order)
        for j in range(batch_size):
            train_data[f'mlsPolyLS{poly_order}'][i * batch_size + j] = np.atleast_2d(coeffs)
        coeffs = utils.mtswrapper.rotate_polynomial(coeffs_ws, in_normal, poly_order)
        for j in range(batch_size):
            train_data[f'mlsPolyTS{poly_order}'][i * batch_size + j] = np.atleast_2d(coeffs)
        coeffs = utils.mtswrapper.rotate_polynomial_azimuth(coeffs_ws, -in_dir, in_normal, poly_order)
        for j in range(batch_size):
            train_data[f'mlsPolyAS{poly_order}'][i * batch_size + j] = np.atleast_2d(coeffs)

        # Additionally extract the local point neighborhood
        if gather_point_cloud:
            pts, nors, weights = get_local_points(
                train_data['inPos'][i * batch_size][None, :], result['inPos'][i], batch_size, n_points, current_g[i], current_albedo[i][0], constraint_kdtree)
            train_data['points'][(i * batch_size):((i + 1) * batch_size)] = pts
            train_data['pointNormals'][(i * batch_size):((i + 1) * batch_size)] = nors
            train_data['pointWeights'][(i * batch_size):((i + 1) * batch_size)] = weights

    # Save out this current train data as pickled file to a subdirectiory
    n_usable = train_data['inPos'].shape[0] // batch_size * batch_size
    for k in train_data:
        train_data[k] = train_data[k][:n_usable, :]

    return train_data


def sample_scattering_3d(mesh_file, mesh_idx, n_train_samples, batch_size, n_abs_samples, medium_param_generator, extra_options):

    ignore_zero_scatter = extra_options['ignore_zero_scatter'] if 'ignore_zero_scatter' in extra_options else IGNORE_ZERO_SCATTER
    compute_sh_coefficients = extra_options['compute_sh_coefficients'] if 'compute_sh_coefficients' in extra_options else False

    seed = extra_options['seed'] if 'seed' in extra_options else 123
    kd_tree_threshold = extra_options['kd_tree_threshold'] if 'kd_tree_threshold' in extra_options else FIT_KDTREE_THRESHOLD
    use_similarity_kernel = extra_options['use_similarity_kernel'] if 'use_similarity_kernel' in extra_options else True
    global_constraint_weight = extra_options['global_constraint_weight'] if 'global_constraint_weight' in extra_options else 0.01

    constraint_kdtree, _, _ = build_constraint_kdtree(mesh_file)

    nParams = n_train_samples // batch_size
    media = medium_param_generator.get_media(nParams)
    current_albedo = media['albedo']
    current_g = media['g']
    current_eta = media['eta']
    current_sigma_t = media['sigmat']
    result = vae.datahandler.sample_scattering_mesh(mesh_file, n_train_samples, batch_size, n_abs_samples, current_albedo, current_sigma_t,
                                                    current_g, current_eta, ignore_zero_scatter, None,
                                                    constraint_kdtree, seed=seed, kd_tree_threshold=kd_tree_threshold, compute_sh_coefficients=compute_sh_coefficients,
                                                    use_similarity_kernel=use_similarity_kernel, global_constraint_weight=global_constraint_weight)

    n_gen_samples = len(result['inPos'])
    print('Done sampling')
    print('NSamples: {}'.format(n_gen_samples))
    train_data = mts_output_to_dict(result)
    train_data['meshIdx'] = np.ones((n_gen_samples, 1), dtype=np.int32) * mesh_idx
    train_data['effAlbedo'] = vae.utils.albedo_to_effective_albedo(
        train_data['albedo'])[..., 0][..., np.newaxis]
    train_data['albedop'] = vae.utils.get_alphap(
        train_data['albedo'], train_data['g'], train_data['sigmaT'])[..., 0][..., np.newaxis]
    train_data.pop('absorptionProbVar', None)

    n_points = 64
    train_data['points'] = np.zeros((n_gen_samples, n_points, 3), dtype=np.float32)
    train_data['pointNormals'] = np.zeros((n_gen_samples, n_points, 3), dtype=np.float32)
    train_data['pointWeights'] = np.zeros((n_gen_samples, n_points), dtype=np.float32)
    for i in range(int(train_data['inPos'].shape[0] // batch_size)):
        pts, nors, weights = get_local_points(train_data['inPos'][i * batch_size][None, :], result['inPos']
                                              [i], batch_size, n_points, current_g[i], current_albedo[i][0], constraint_kdtree)
        train_data['points'][(i * batch_size):((i + 1) * batch_size)] = pts
        train_data['pointNormals'][(i * batch_size):((i + 1) * batch_size)] = nors
        train_data['pointWeights'][(i * batch_size):((i + 1) * batch_size)] = weights

    # Save out this current train data as pickled file to a subdirectiory
    n_usable = train_data['inPos'].shape[0] // batch_size * batch_size
    for k in train_data:
        train_data[k] = train_data[k][:n_usable, :]

    return train_data


def gaussian_data_gen(n_train_samples, batch_size):
    train_data = {}
    train_data['inPos'] = np.zeros((n_train_samples, 3))
    train_data['inDir'] = np.zeros((n_train_samples, 3))
    train_data['inDir'][:, 1] = 1
    train_data['outPos'] = np.random.randn(n_train_samples, 3)
    train_data['outDir'] = np.random.randn(n_train_samples, 3)
    train_data['inNormal'] = np.zeros((n_train_samples, 3))
    train_data['outNormal'] = np.zeros((n_train_samples, 3))
    train_data['outNormal'][:, 1] = 1
    train_data['inNormal'][:, 1] = 1
    train_data['throughput'] = np.zeros((n_train_samples, 3))
    train_data['absorptionProb'] = np.ones((n_train_samples, 1)) * 0.5
    train_data['albedo'] = np.ones((n_train_samples, 3)) * 0.5
    train_data['effAlbedo'] = np.ones((n_train_samples, 1)) * 0.5
    train_data['sigmaT'] = np.ones((n_train_samples, 3)) * 0.5
    train_data['g'] = np.ones((n_train_samples, 1)) * 0.5
    train_data['bounces'] = np.ones((n_train_samples, 1), dtype=np.int32)
    train_data['ior'] = np.ones((n_train_samples, 1)) * 0.5
    train_data['mlsPoly3'] = np.ones((n_train_samples, 20)) * 0.5
    train_data['mlsPolyLS3'] = np.ones((n_train_samples, 20)) * 0.5
    train_data['meshIdx'] = np.ones((n_train_samples, 1), dtype=np.int32) * 0.5
    for k in train_data:
        if train_data[k].dtype == np.float64:
            train_data[k] = train_data[k].astype(np.float32)
        elif train_data[k].dtype == np.int64:
            train_data[k] = train_data[k].astype(np.int32)

    # Save out this current train data as pickled file to a subdirectiory
    n_usable = train_data['inPos'].shape[0] // batch_size * batch_size
    for k in train_data:
        train_data[k] = train_data[k][:n_usable, :]
    return train_data


def process_sampled_data(batch_size, traindata_file, constant_variables=[]):
    # pickle_files = glob.glob(traindata_file + '/data_*[0-9].pickle'.format(poly_order))
    print('Finding files in: {}'.format(traindata_file))
    pickle_files = glob.glob(traindata_file + '/data_*[0-9]_*[0-9].pickle')
    random.shuffle(pickle_files)
    # iterate through all the files and concatenate them, write to disk
    idx = 0
    n_samples = []
    stats = []
    with open(pickle_files[0], 'rb') as f:
        d = pickle.load(f)
        train_data_shard = {k: [] for k in d}

    use_legendre = True
    if use_legendre and 'mlsPolyLS3' in train_data_shard:
        transform = vae.utils.get_legendre_transform(3)
    if use_legendre and 'mlsPolyLS2' in train_data_shard:
        transform = vae.utils.get_legendre_transform(2)

    n_shard_samples = 0
    for pidx, pf in enumerate(tqdm.tqdm(pickle_files)):
        with open(pf, 'rb') as f:
            d = pickle.load(f)

        # append to current train_data_shard
        for k in train_data_shard:
            train_data_shard[k].append(d[k])

        n_shard_samples += d['inPos'].shape[0]
        if n_shard_samples >= 1024000 or pidx == len(pickle_files) - 1:
            train_data = {}
            for k in train_data_shard:
                train_data[k] = np.concatenate(train_data_shard[k], axis=0)

            if 'normalHist' in train_data:
                p = ((train_data['normalHist'] * 2.0 - 1.0) * 0.9 + 1.0) * 0.5
                train_data['normalHist'] = utils.math.logit(p).astype(np.float32)

            if use_legendre and 'mlsPolyLS3' in train_data:
                train_data['legendrePolyLS3'] = ((transform @ train_data['mlsPolyLS3'].T).T).astype(np.float32)
            if use_legendre and 'mlsPolyLS2' in train_data:
                train_data['legendrePolyLS2'] = ((transform @ train_data['mlsPolyLS2'].T).T).astype(np.float32)

            shard_stats, n_samps = vae.datahandler.get_stats_from_dict(train_data, constant_variables)
            n_samples.append(n_samps)
            stats.append(shard_stats)

            with tf.python_io.TFRecordWriter(traindata_file + '/data_{:03d}.tfrecord'.format(idx)) as writer:
                for i in range(train_data['inPos'].shape[0] // batch_size):
                    feat_dict = {k: bytes_feature(train_data[k][(i * batch_size):(i + 1)
                                                                * batch_size].tostring()) for k in train_data}
                    example = tf.train.Example(features=tf.train.Features(feature=feat_dict))
                    writer.write(example.SerializeToString())
            idx += 1
            n_shard_samples = 0
            for k in train_data_shard:
                train_data_shard[k] = []
    # Average stats and save them out to pickle and json
    weights = np.array(n_samples)
    weights = weights / np.sum(weights)
    stats2 = {}
    for k in stats[0].keys():
        mean = np.zeros_like(stats[0][k])
        for i in range(len(stats)):
            mean += weights[i] * stats[i][k]
        stats2[k] = mean

    stats2['n_samples'] = int(np.sum(n_samples))
    stats2['n_batches'] = int(np.sum(n_samples) // batch_size)
    with open(traindata_file + '/data_stats.pickle', 'wb') as f:
        pickle.dump(stats2, f)
    for k in stats2:
        if type(stats2[k]) == np.ndarray:
            stats2[k] = stats2[k].tolist()
    with open(traindata_file + '/data_stats.json', 'w') as f:
        json.dump(stats2, f, sort_keys=False, indent=4, ensure_ascii=False)


class MeshGenerator:
    def __init__(self, scene_dir, resource_dir, mesh_folder='complexshapes', name='default'):
        self.test_fraction = 0.2
        self.n_scenes = 200
        self.name = name
        self.scene_dir = scene_dir
        self.resource_dir = resource_dir
        self.mesh_folder = mesh_folder

    def random_meshes(self, n_meshes, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        mesh_files = []
        for i in range(n_meshes):
            mesh_out_path = os.path.join(output_dir, 'mesh{:03d}.obj'.format(i))
            mesh_files.append(mesh_out_path)
            if os.path.isfile(mesh_out_path):
                continue
            used_shape_path = np.random.choice(glob.glob(self.resource_dir + '/' + self.mesh_folder + '/*.obj'))
            mesh = trimesh.load_mesh(used_shape_path, process=False)

            # Scale mesh to [-10, 10] ** 3
            bounds = mesh.bounds
            mesh.apply_translation(-bounds[0])
            max_diag = np.max(bounds[1] - bounds[0])
            mesh.apply_scale(1 / max_diag)
            mesh.apply_translation(-np.ones(3) * 0.5)
            bounds = mesh.bounds
            mean = 0.5 * (bounds[0] + bounds[1])
            mesh.apply_translation(-mean)

            # randomly rotate and scale mesh
            rotate = trimesh.transformations.random_rotation_matrix(np.random.rand(3))
            mesh.apply_transform(rotate)
            # Constant scale which usually gives interesting results/level of detail for sigmaT = 1.0
            mesh.apply_scale(20)

            random_scale = np.random.rand(4) * 2.75 + 0.25
            # random_scale = np.random.rand(4) * 2.0 + 1.0
            random_scale[3] = 1
            mesh.apply_transform(np.diag(random_scale))
            with open(mesh_out_path, 'w') as obj_file:
                obj_file.write(trimesh.io.wavefront.export_wavefront(mesh))
        return mesh_files

    def get_meshes(self, generate=True):
        if generate:
            n_train_fraction = int(self.n_scenes * 0.8)
            train_meshes = self.random_meshes(n_train_fraction, os.path.join(self.scene_dir, self.name, 'train'))
            test_meshes = self.random_meshes(self.n_scenes - n_train_fraction,
                                             os.path.join(self.scene_dir, self.name, 'test'))
        else:
            train_meshes = sorted(glob.glob(os.path.join(self.scene_dir, self.name, 'train') + '/*.obj'))
            test_meshes = sorted(glob.glob(os.path.join(self.scene_dir, self.name, 'test') + '/*.obj'))

        return train_meshes, test_meshes


class SphereMeshGenerator(MeshGenerator):
    def __init__(self, scene_dir, resource_dir):
        super().__init__(scene_dir, resource_dir)
        self.name = 'sphere'
        self.scale = 1.0

    def get_meshes(self, generate=True):
        if not generate:
            p = os.path.join(self.scene_dir, self.name, 'sphere.obj')
            return [p], [p]
        out_dir = os.path.join(self.scene_dir, self.name)
        os.makedirs(out_dir, exist_ok=True)
        mesh_out_path = os.path.join(out_dir, 'sphere.obj')
        mesh_files = [mesh_out_path]
        used_shape_path = self.resource_dir + '/baseshapes/sphere.obj'
        mesh = trimesh.load_mesh(used_shape_path, process=False)
        bounds = mesh.bounds
        mesh.apply_translation(-bounds[0])
        max_diag = np.max(bounds[1] - bounds[0])
        mesh.apply_scale(1 / max_diag)
        mesh.apply_translation(-np.ones(3) * 0.5)
        bounds = mesh.bounds
        mean = 0.5 * (bounds[0] + bounds[1])
        mesh.apply_translation(-mean)
        mesh.apply_scale(20 * self.scale)
        with open(mesh_out_path, 'w') as obj_file:
            obj_file.write(trimesh.io.wavefront.export_wavefront(mesh))
        return mesh_files, mesh_files


class BigSphereMeshGenerator(SphereMeshGenerator):
    def __init__(self, scene_dir, resource_dir):
        super().__init__(scene_dir, resource_dir)
        self.name = 'bigsphere'
        self.scale = 3.0


class ScaledSphereMeshGenerator(MeshGenerator):
    def __init__(self, scene_dir, resource_dir):
        super().__init__(scene_dir, resource_dir)
        self.name = 'scaledsphere'
        self.test_fraction = 0.2
        self.n_scenes = 200

    def sphere_meshes(self, n_meshes, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        mesh_files = []
        for i in range(n_meshes):
            mesh_out_path = os.path.join(output_dir, 'mesh{:03d}.obj'.format(i))
            mesh_files.append(mesh_out_path)
            if os.path.isfile(mesh_out_path):
                continue
            used_shape_path = self.resource_dir + '/baseshapes/sphere.obj'
            mesh = trimesh.load_mesh(used_shape_path, process=False)

            # Scale mesh to [-10, 10] ** 3
            bounds = mesh.bounds
            mesh.apply_translation(-bounds[0])
            max_diag = np.max(bounds[1] - bounds[0])
            mesh.apply_scale(1 / max_diag)
            mesh.apply_translation(-np.ones(3) * 0.5)
            bounds = mesh.bounds
            mean = 0.5 * (bounds[0] + bounds[1])
            mesh.apply_translation(-mean)
            mesh.apply_scale(20)

            random_scale = np.ones(4) * np.random.rand(1) * 2.75 + 0.25
            random_scale[3] = 1
            mesh.apply_transform(np.diag(random_scale))
            with open(mesh_out_path, 'w') as obj_file:
                obj_file.write(trimesh.io.wavefront.export_wavefront(mesh))
        return mesh_files

    def get_meshes(self, generate=True):
        if generate:
            n_train_fraction = int(self.n_scenes * 0.8)
            train_meshes = self.sphere_meshes(n_train_fraction, os.path.join(self.scene_dir, self.name, 'train'))
            test_meshes = self.sphere_meshes(self.n_scenes - n_train_fraction,
                                             os.path.join(self.scene_dir, self.name, 'test'))
        else:
            train_meshes = sorted(glob.glob(os.path.join(self.scene_dir, self.name, 'train') + '/*.obj'))
            test_meshes = sorted(glob.glob(os.path.join(self.scene_dir, self.name, 'test') + '/*.obj'))

        return train_meshes, test_meshes


class PlaneMeshGenerator(MeshGenerator):
    def __init__(self, scene_dir, resource_dir):
        super().__init__(scene_dir, resource_dir)
        self.name = 'plane'

    def get_meshes(self, generate=True):
        m = os.path.join(self.resource_dir, 'test', 'ground.obj')
        return [m], [m]


def save_mesh_sample_stats(mesh_idx, duration, train_data, output_dir):
    d = {}
    d['mesh'] = mesh_idx
    d['time'] = duration
    d['min_albedo'] = float(np.min(train_data['albedo']))
    d['n_samples'] = int(train_data['albedo'].shape[0])
    with open(output_dir + f'/stats_{mesh_idx:06d}.json', 'w') as f:
        json.dump(d, f)


def sample(mesh_name, mesh_idx, batch_size, n_abs_samples, medium_param_generator, n_samples, extra_options):
    seed = extra_options['seed'] if 'seed' in extra_options else 123
    np.random.seed(seed)
    if 'true_scattering' in extra_options and extra_options['true_scattering']:
        return sample_scattering_3d(mesh_name, mesh_idx, n_samples, batch_size, n_abs_samples,
                                    medium_param_generator, extra_options)
    else:
        return sample_scattering_poly_3d(mesh_name, mesh_idx, n_samples, batch_size, n_abs_samples,
                                         medium_param_generator, extra_options)


def sample_and_save(mesh_name, mesh_idx, output_dir, batch_size, n_abs_samples, medium_param_generator,
                    n_samples, shard_size, batch_reduction_factor, extra_options):
    t0 = time.time()
    train_data = sample(mesh_name, mesh_idx, batch_size, n_abs_samples,
                        medium_param_generator,  n_samples * batch_reduction_factor, extra_options)
    duration = time.time() - t0
    print(f'Took {duration} s')
    for k in train_data.keys():
        train_data[k] = train_data[k][::batch_reduction_factor, :]
    save_mesh_sample_stats(mesh_idx, duration, train_data, output_dir)
    save_sharded(mesh_idx, train_data, shard_size, output_dir)


class ScatterData:

    def __init__(self, scene_dir, resource_dir, dataset_folder):
        self.mesh_generator = MeshGenerator(scene_dir, resource_dir)
        self.medium_param_generator = MixedMediumParamGenerator()

        self.dataset_folder = dataset_folder
        self.n_train_samples = 100000
        self.constant_variables = []
        self.searchlight_configuration = False
        self.importance_sample_polynomials = False
        self.gather_point_cloud = False
        self.compute_sh_coefficients = False

        # What fraction of the batch do we actually keep

        self.batch_size = 16
        self.n_abs_samples = 1024
        self.batch_reduction_factor = 1
        self.shard_size = 20 * 128
        self.use_true_scattering = False

        self.polyfit_config = utils.mtswrapper.PolyFitConfig()
        self.polyfit_config.order = 3
        self.polyfit_config.use_similarity_kernel = True
        self.polyfit_config.global_constraints_weight = 0.01

        self.compute_normal_histogram = False


    def get_polyfit_config_dict(self):
        return utils.mtswrapper.poly_fit_config_to_dict(self.polyfit_config)

    def test_path(self):
        return os.path.join(self.dataset_folder, 'test')

    def train_path(self):
        return os.path.join(self.dataset_folder, 'train')

    def generate_meshes(self):
        self.mesh_generator.get_meshes()

    def generate(self, n_threads=12, cluster_job=False, job_id=5, n_jobs=10):
        os.makedirs(self.train_path(), exist_ok=True)
        os.makedirs(self.test_path(), exist_ok=True)
        train_meshes, test_meshes = self.mesh_generator.get_meshes(not cluster_job)
        m = [(idx, mesh, self.train_path()) for idx, mesh in enumerate(train_meshes)]
        m += [(idx, mesh, self.test_path()) for idx, mesh in enumerate(test_meshes)]

        if cluster_job:  # Process only part of the given meshes
            meshes_per_job = len(m) // n_jobs
            if job_id == n_jobs - 1:
                m_new = m[(meshes_per_job * job_id):]
            else:
                m_new = m[(meshes_per_job * job_id):(meshes_per_job * (job_id + 1))]
            m = m_new
        # Its better to parallelize over samples than meshes if there are very few meshes
        parallelize_each_mesh = len(m) < n_threads
        p = multiprocessing.Pool(n_threads)
        extra_options = {'searchlight_configuration': self.searchlight_configuration, 'importance_sample_polynomials': self.importance_sample_polynomials,
                         'gather_point_cloud': self.gather_point_cloud, 'true_scattering': self.use_true_scattering, 'compute_sh_coefficients': self.compute_sh_coefficients,
                         'compute_normal_histogram': self.compute_normal_histogram}
        extra_options.update(self.get_polyfit_config_dict())

        if parallelize_each_mesh:
            n_batches = self.batch_reduction_factor * self.n_train_samples // self.batch_size
            sample_counts = n_batches // n_threads
            sample_counts = [sample_counts] * n_threads
            sample_counts[-1] += n_batches % n_threads
            sample_counts = [s * self.batch_size for s in sample_counts]
            for t in m:
                idx, mesh, output_dir = t[0], t[1], t[2]
                seed = int(np.random.randint(10000))
                arguments = []
                for i in range(n_threads):
                    extra_options['seed'] = seed + i
                    arguments.append(tuple([mesh, idx, self.batch_size, self.n_abs_samples,
                                            self.medium_param_generator, sample_counts[i], extra_options]))
                t0 = time.time()
                res = p.starmap(sample, arguments)
                duration = time.time() - t0
                print(f'Took {duration} s')
                train_data = {}
                for k in res[0].keys():
                    train_data[k] = np.concatenate([res[i][k] for i in range(len(res))], axis=0)

                # Remove some samples from each batch
                for k in train_data.keys():
                    train_data[k] = train_data[k][::self.batch_reduction_factor, :]
                save_mesh_sample_stats(idx, duration, train_data, output_dir)
                save_sharded(idx, train_data, self.shard_size, output_dir)
        else:
            # parallelize across meshes: Each mesh in a separate process
            arguments = []
            for t in m:
                idx, mesh, output_dir = t[0], t[1], t[2]
                seed = int(np.random.randint(10000))
                extra_options['seed'] = seed
                arguments.append(tuple([mesh, idx, output_dir, self.batch_size, self.n_abs_samples,
                                        self.medium_param_generator, self.n_train_samples, self.shard_size,
                                        self.batch_reduction_factor, extra_options]))
            p.starmap(sample_and_save, arguments)

    def process(self):
        t0 = time.time()
        process_sampled_data(self.batch_size // self.batch_reduction_factor,
                             self.train_path(), self.constant_variables)
        process_sampled_data(self.batch_size // self.batch_reduction_factor,
                             self.test_path(), self.constant_variables)
        print(f'TFRecord saving and stats took {time.time() - t0} s')

    def get_feature_statistics(self):
        return vae.utils.get_feature_stats(self.train_path() + '/data')

    def get_dataset_iterator(self, batch_size, numepochs, config, feature_statistics, absorption_only):
        traindata_files = glob.glob(self.train_path() + '/data_[0-9][0-9][0-9].tfrecord')
        testdata_files = glob.glob(self.test_path() + '/data_[0-9][0-9][0-9].tfrecord')
        if len(traindata_files) == 0:
            print('Error: Train data not found')
            quit()
        if len(testdata_files) == 0:
            print('Error: Test data not found')
            quit()

        it_train = vae.datahandler.get_dataset_iterator(
            traindata_files, batch_size, numepochs, config, feature_statistics, absorption_only)
        it_test = vae.datahandler.get_dataset_iterator(
            testdata_files, batch_size, 10 * numepochs, config, feature_statistics, absorption_only, test=True)
        return it_train, it_test


class ScatterDataSimilarity(ScatterData):
    def __init__(self, scene_dir, resource_dir, data_dir):
        super().__init__(scene_dir, resource_dir, data_dir)
        self.medium_param_generator = SimilarityMediumParamGenerator()
        self.polyfit_config.use_similarity_kernel = True
        self.polyfit_config.global_constraints_weight = 0.01
        # self.regularization = 4.0


class ScatterDataSimilarityMixed(ScatterData):
    def __init__(self, scene_dir, resource_dir, data_dir):
        super().__init__(scene_dir, resource_dir, data_dir)
        self.medium_param_generator = MixedMediumParamGenerator()
        self.polyfit_config.use_similarity_kernel = True
        self.polyfit_config.global_constraints_weight = 0.01
        # self.regularization = 4.0



class ScatterDataSlabMesh(ScatterData):
    def __init__(self, scene_dir, resource_dir, data_dir):
        super().__init__(scene_dir, resource_dir, data_dir)
        self.mesh_generator = MeshGenerator(scene_dir, resource_dir, 'slab', name='slab')


class ScatterDataSlabs(ScatterDataSimilarityMixed):
    def __init__(self, scene_dir, resource_dir, data_dir):
        super().__init__(scene_dir, resource_dir, data_dir)
        self.n_train_samples = 1000

    def sample_scattering_poly_3d_simple(self, n_train_samples, batch_size, n_abs_samples,
                                medium_param_generator, extra_options):

        poly_order = extra_options['order'] if 'order' in extra_options else 3
        seed = extra_options['seed'] if 'seed' in extra_options else 123
        fixed_polynomial = extra_options['fixed_polynomial']
        nParams = n_train_samples // batch_size
        media = medium_param_generator.get_media(nParams)
        result = utils.mtswrapper.sample_scattering_poly(
            n_train_samples, batch_size, n_abs_samples, media, fixed_polynomial, seed, extra_options)
        n_gen_samples = len(result['inPos'])
        print('Done sampling')
        print('NSamples: {}'.format(n_gen_samples))
        train_data = mts_output_to_dict(result, {'shapeCoeffs': f'mlsPoly{poly_order}'})
        train_data['meshIdx'] = np.ones((n_gen_samples, 1), dtype=np.int32)
        train_data['effAlbedo'] = vae.utils.albedo_to_effective_albedo(train_data['albedo'])[..., 0][..., np.newaxis]
        train_data['albedop'] = vae.utils.get_alphap(train_data['albedo'], train_data['g'], train_data['sigmaT'])[..., 0][..., np.newaxis]
        train_data.pop('absorptionProbVar', None)
        train_data[f'mlsPolyLS{poly_order}'] = np.zeros_like(train_data[f'mlsPoly{poly_order}'])
        train_data[f'mlsPolyTS{poly_order}'] = np.zeros_like(train_data[f'mlsPoly{poly_order}'])
        train_data[f'mlsPolyAS{poly_order}'] = np.zeros_like(train_data[f'mlsPoly{poly_order}'])
        for i in range(int(train_data['inPos'].shape[0] // batch_size)):
            coeffs_ws = np.array(train_data[f'mlsPoly{poly_order}'][i * batch_size])
            in_dir = train_data['inDir'][i * batch_size]
            in_normal = train_data['inNormal'][i * batch_size]
            coeffs = utils.mtswrapper.rotate_polynomial(coeffs_ws, -in_dir, poly_order)
            for j in range(batch_size):
                train_data[f'mlsPolyLS{poly_order}'][i * batch_size + j] = np.atleast_2d(coeffs)
            coeffs = utils.mtswrapper.rotate_polynomial(coeffs_ws, in_normal, poly_order)
            for j in range(batch_size):
                train_data[f'mlsPolyTS{poly_order}'][i * batch_size + j] = np.atleast_2d(coeffs)
            coeffs = utils.mtswrapper.rotate_polynomial_azimuth(coeffs_ws, -in_dir, in_normal, poly_order)
            for j in range(batch_size):
                train_data[f'mlsPolyAS{poly_order}'][i * batch_size + j] = np.atleast_2d(coeffs)
        # Save out this current train data as pickled file to a subdirectiory
        n_usable = train_data['inPos'].shape[0] // batch_size * batch_size
        for k in train_data:
            train_data[k] = train_data[k][:n_usable, :]
        return train_data

    def generate(self, n_threads=12, cluster_job=False, job_id=5, n_jobs=10):
        os.makedirs(self.train_path(), exist_ok=True)
        os.makedirs(self.test_path(), exist_ok=True)

        extra_options = {'searchlight_configuration': self.searchlight_configuration, 'importance_sample_polynomials': False,
                         'gather_point_cloud': self.gather_point_cloud}
        extra_options.update(self.get_polyfit_config_dict())

        for total_samples, output_dir in [(to_multiple_of(self.n_train_samples * 0.8, self.batch_size),
                                           self.train_path()), (to_multiple_of(self.n_train_samples * 0.2, self.batch_size), self.test_path())]:

            n_batches = self.batch_reduction_factor * total_samples // self.batch_size
            sample_counts = n_batches
            coeffs = np.zeros(20, dtype=np.float32)
            coeffs[2] = 1.0

            # coeffs = np.zeros((1, 20) ,dtype=np.float32)
            # coeffs[:, 3] = 1.0 / scale_factor
            # coeffs[:, 9] = 1.0 / scale_factor**2

            extra_options['seed'] = int(np.random.randint(10000))
            extra_options['fixed_polynomial'] = coeffs
            t0 = time.time()
            train_data = self.sample_scattering_poly_3d_simple(self.n_train_samples, self.batch_size, self.n_abs_samples,
                                self.medium_param_generator, extra_options)
            duration = time.time() - t0
            print(f'Took {duration} s')

            # Remove some samples from each batch
            for k in train_data.keys():
                train_data[k] = train_data[k][::self.batch_reduction_factor, :]
            save_mesh_sample_stats(0, duration, train_data, output_dir)
            save_sharded(0, train_data, self.shard_size, output_dir)




class ScatterDataMixed(ScatterData):
    def __init__(self, scene_dir, resource_dir, data_dir):
        super().__init__(scene_dir, resource_dir, data_dir)
        self.medium_param_generator = MixedMediumParamGenerator()
        self.use_similarity_kernel = False
        self.global_constraints_weight = 0.01

class ScatterDataMixed2(ScatterData):
    def __init__(self, scene_dir, resource_dir, data_dir):
        super().__init__(scene_dir, resource_dir, data_dir)
        self.medium_param_generator = MixedMediumParamGenerator2()

class ScatterDataMixed3(ScatterData):
    def __init__(self, scene_dir, resource_dir, data_dir):
        super().__init__(scene_dir, resource_dir, data_dir)
        self.medium_param_generator = MixedMediumParamGenerator3()

class ScatterDataMixed4(ScatterData):
    def __init__(self, scene_dir, resource_dir, data_dir):
        super().__init__(scene_dir, resource_dir, data_dir)
        self.medium_param_generator = MixedMediumParamGenerator4()


class ScatterDataMixedNoGlobConstraint(ScatterData):
    def __init__(self, scene_dir, resource_dir, data_dir):
        super().__init__(scene_dir, resource_dir, data_dir)
        self.medium_param_generator = MixedMediumParamGenerator()
        self.use_similarity_kernel = False
        self.global_constraints_weight = 0.0


class ScatterDataHardConstraint(ScatterData):
    def __init__(self, scene_dir, resource_dir, data_dir):
        super().__init__(scene_dir, resource_dir, data_dir)
        self.use_hard_surface_constraint = True


class ScatterDataNormalHist(ScatterData):
    def __init__(self, scene_dir, resource_dir, data_dir):
        super().__init__(scene_dir, resource_dir, data_dir)
        self.compute_normal_histogram = True


class ScatterDataPcTrue(ScatterData):

    def __init__(self, scene_dir, resource_dir, data_dir):
        super().__init__(scene_dir, resource_dir, data_dir)
        self.n_train_samples = 100000
        self.gather_point_cloud = True
        self.n_abs_samples = 1024
        self.kd_tree_threshold = 0.0
        self.use_true_scattering = True


class ScatterDataPcShTrue(ScatterData):
    def __init__(self, scene_dir, resource_dir, data_dir):
        super().__init__(scene_dir, resource_dir, data_dir)
        self.n_train_samples = 100000
        self.gather_point_cloud = True
        self.n_abs_samples = 1024
        self.kd_tree_threshold = 0.0
        self.use_true_scattering = True
        self.compute_sh_coefficients = True


class ScatterDataHighAbs(ScatterData):
    def __init__(self, scene_dir, resource_dir, data_dir):
        super().__init__(scene_dir, resource_dir, data_dir)
        self.n_abs_samples = 1024
        self.kd_tree_threshold = 0.0


class ScatterDataHighAbsDeg2(ScatterDataHighAbs):
    def __init__(self, scene_dir, resource_dir, data_dir):
        super().__init__(scene_dir, resource_dir, data_dir)
        self.poly_order = 2


class ScatterDataIs(ScatterData):
    def __init__(self, scene_dir, resource_dir, data_dir):
        super().__init__(scene_dir, resource_dir, data_dir)
        self.importance_sample_polynomials = True


class ScatterDataNoThreshold(ScatterData):
    def __init__(self, scene_dir, resource_dir, data_dir):
        super().__init__(scene_dir, resource_dir, data_dir)
        self.kd_tree_threshold = 0.0


class ScatterDataNoThresholdDeg2(ScatterDataNoThreshold):
    def __init__(self, scene_dir, resource_dir, data_dir):
        super().__init__(scene_dir, resource_dir, data_dir)
        self.poly_order = 2


class ScatterDataMoreReg(ScatterData):
    def __init__(self, scene_dir, resource_dir, data_dir):
        super().__init__(scene_dir, resource_dir, data_dir)
        self.regularization = 1e-1
        self.kd_tree_threshold = 0.0


class ScatterDataHighReg(ScatterData):
    def __init__(self, scene_dir, resource_dir, data_dir):
        super().__init__(scene_dir, resource_dir, data_dir)
        self.regularization = 10.0
        self.kd_tree_threshold = 0.0


class ScatterDataHighReg2(ScatterData):
    def __init__(self, scene_dir, resource_dir, data_dir):
        super().__init__(scene_dir, resource_dir, data_dir)
        self.regularization = 50.0
        self.kd_tree_threshold = 0.0


class ScatterDataHighRegDeg2(ScatterDataHighReg):
    def __init__(self, scene_dir, resource_dir, data_dir):
        super().__init__(scene_dir, resource_dir, data_dir)
        self.poly_order = 2


class ScatterDataHighReg2Deg2(ScatterDataHighReg2):
    def __init__(self, scene_dir, resource_dir, data_dir):
        super().__init__(scene_dir, resource_dir, data_dir)
        self.poly_order = 2


class ScatterDataPc(ScatterData):
    def __init__(self, scene_dir, resource_dir, data_dir):
        super().__init__(scene_dir, resource_dir, data_dir)
        self.gather_point_cloud = True


class ScatterDataDeg2(ScatterData):
    def __init__(self, scene_dir, resource_dir, data_dir):
        super().__init__(scene_dir, resource_dir, data_dir)
        self.poly_order = 2


class ScatterDataSmall(ScatterData):
    def __init__(self, scene_dir, resource_dir, data_dir):
        super().__init__(scene_dir, resource_dir, data_dir)
        self.batch_reduction_factor = 1


class ScatterDataSphere(ScatterData):

    def __init__(self, scene_dir, resource_dir, data_dir):
        super().__init__(scene_dir, resource_dir, data_dir)
        self.mesh_generator = SphereMeshGenerator(scene_dir, resource_dir)
        self.n_train_samples = 1000000


class ScatterDataSpherePc(ScatterDataSphere):

    def __init__(self, scene_dir, resource_dir, data_dir):
        super().__init__(scene_dir, resource_dir, data_dir)
        self.gather_point_cloud = True


class ScatterDataSphereDeg2(ScatterDataSphere):

    def __init__(self, scene_dir, resource_dir, data_dir):
        super().__init__(scene_dir, resource_dir, data_dir)
        self.poly_order = 2


class ScatterDataSphereDeg2Pc(ScatterDataSphereDeg2):
    def __init__(self, scene_dir, resource_dir, data_dir):
        super().__init__(scene_dir, resource_dir, data_dir)
        self.gather_point_cloud = True


class ScatterDataSphereScaled(ScatterData):

    def __init__(self, scene_dir, resource_dir, data_dir):
        super().__init__(scene_dir, resource_dir, data_dir)
        self.mesh_generator = ScaledSphereMeshGenerator(scene_dir, resource_dir)
        self.n_train_samples = 100000


class ScatterDataSphereFixedMedium(ScatterData):

    def __init__(self, scene_dir, resource_dir, data_dir):
        super().__init__(scene_dir, resource_dir, data_dir)
        self.mesh_generator = SphereMeshGenerator(scene_dir, resource_dir)
        self.n_train_samples = 1000000
        self.constant_variables = ['g', 'albedo', 'effAlbedo', 'ior']
        self.medium_param_generator = FixedMediumGenerator()


class ScatterDataBigSphereFixedMedium(ScatterData):

    def __init__(self, scene_dir, resource_dir, data_dir):
        super().__init__(scene_dir, resource_dir, data_dir)
        self.mesh_generator = BigSphereMeshGenerator(scene_dir, resource_dir)
        self.n_train_samples = 1000000
        self.constant_variables = ['g', 'albedo', 'effAlbedo', 'ior']
        self.medium_param_generator = FixedMediumGenerator()


class ScatterDataSphereFixedMediumDeg2(ScatterDataSphereFixedMedium):
    def __init__(self, scene_dir, resource_dir, data_dir):
        super().__init__(scene_dir, resource_dir, data_dir)
        self.poly_order = 2


class ScatterDataSphereFixedMediumSearchlight(ScatterData):

    def __init__(self, scene_dir, resource_dir, data_dir):
        super().__init__(scene_dir, resource_dir, data_dir)
        self.mesh_generator = SphereMeshGenerator(scene_dir, resource_dir)
        self.n_train_samples = 1000000
        self.constant_variables = ['g', 'albedo', 'effAlbedo', 'ior']
        self.medium_param_generator = FixedMediumGenerator()
        self.searchlight_configuration = True


class ScatterDataSphereFixedMediumSearchlightDeg2(ScatterDataSphereFixedMediumSearchlight):
    def __init__(self, scene_dir, resource_dir, data_dir):
        super().__init__(scene_dir, resource_dir, data_dir)
        self.poly_order = 2


def to_multiple_of(num, divisor):
    num = int(num)
    return (divisor + 1) * (num // divisor) if num % divisor != 0 else num


class ScatterDataPlaneFixedMediumSearchlight(ScatterData):

    def __init__(self, scene_dir, resource_dir, data_dir):
        super().__init__(scene_dir, resource_dir, data_dir)
        self.poly_order = 1
        # No meshes need to be generated since we assume an infinite plane, but function expects a list of meshes
        self.mesh_generator = PlaneMeshGenerator(scene_dir, resource_dir)
        self.n_train_samples = 1000000
        self.constant_variables = ['g', 'albedo', 'effAlbedo', 'ior',
                                   'outPosY', 'outPosRelWSY', 'outPosRelTSY', 'outPosRelLSZ']
        self.medium_param_generator = FixedMediumGenerator()
        self.searchlight_configuration = True
        self.batch_reduction_factor = 1

    def generate(self, n_threads=12, cluster_job=False, job_id=5, n_jobs=10):
        os.makedirs(self.train_path(), exist_ok=True)
        os.makedirs(self.test_path(), exist_ok=True)

        coeffs = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)
        extra_options = {'searchlight_configuration': self.searchlight_configuration, 'importance_sample_polynomials': False,
                         'gather_point_cloud': self.gather_point_cloud, 'order': self.poly_order}

        for total_samples, output_dir in [(to_multiple_of(self.n_train_samples * 0.8, self.batch_size),
                                           self.train_path()), (to_multiple_of(self.n_train_samples * 0.2, self.batch_size), self.test_path())]:
            p = multiprocessing.Pool(n_threads)
            n_batches = self.batch_reduction_factor * total_samples // self.batch_size
            sample_counts = n_batches // n_threads
            sample_counts = [sample_counts] * n_threads
            sample_counts[-1] += n_batches % n_threads
            sample_counts = [s * self.batch_size for s in sample_counts]

            seed = int(np.random.randint(10000))
            arguments = []
            for i in range(n_threads):
                extra_options['seed'] = seed + i
                extra_options['fixed_polynomial'] = coeffs
                arguments.append(tuple([None, 0, self.batch_size, self.n_abs_samples,
                                        self.medium_param_generator, sample_counts[i], extra_options]))
            t0 = time.time()
            res = p.starmap(sample, arguments)
            duration = time.time() - t0
            print(f'Took {duration} s')
            train_data = {}
            for k in res[0].keys():
                train_data[k] = np.concatenate([res[i][k] for i in range(len(res))], axis=0)

            # Remove some samples from each batch
            for k in train_data.keys():
                train_data[k] = train_data[k][::self.batch_reduction_factor, :]
            save_mesh_sample_stats(0, duration, train_data, output_dir)
            save_sharded(0, train_data, self.shard_size, output_dir)


class ScatterDataPlane(ScatterData):

    def __init__(self, scene_dir, resource_dir, data_dir):
        super().__init__(scene_dir, resource_dir, data_dir)
        self.poly_order = 1
        # No meshes need to be generated since we assume an infinite plane, but function expects a list of meshes
        self.mesh_generator = PlaneMeshGenerator(scene_dir, resource_dir)
        self.n_train_samples = 1000000

    def sample(self, mesh_name):
        coeffs = np.array([0.0, 0.0, 1.0, 0.0])
        extra_options = {'searchlight_configuration': True, 'fixed_polynomial': coeffs}
        return sample_scattering_poly_3d(mesh_name, self.n_train_samples, self.batch_size,
                                         self.poly_order, self.medium_param_generator, extra_options)


class GaussianData(ScatterData):

    def __init__(self, scene_dir, resource_dir, data_dir):
        super().__init__(scene_dir, resource_dir, data_dir)
        self.poly_order = 2
        # No meshes need to be generated since we assume an infinite plane, but function expects a list of meshes
        self.mesh_generator = PlaneMeshGenerator(scene_dir, resource_dir)
        self.n_train_samples = 1000000
        self.constant_variables = ['g', 'albedo', 'effAlbedo', 'ior']

    def sample(self, mesh_name):
        return gaussian_data_gen(self.n_train_samples, self.batch_size)


class ScatterDataFixedMedium(ScatterData):

    def __init__(self, scene_dir, resource_dir, data_dir):
        super().__init__(scene_dir, resource_dir, data_dir)
        self.constant_variables = ['g', 'albedo', 'effAlbedo', 'ior']
        self.medium_param_generator = FixedMediumGenerator()


class ScatterDataFixedMediumDeg2(ScatterDataFixedMedium):
    def __init__(self, scene_dir, resource_dir, data_dir):
        super().__init__(scene_dir, resource_dir, data_dir)
        self.poly_order = 2


class ScatterDataFixedMediumSearchlight(ScatterData):

    def __init__(self, scene_dir, resource_dir, data_dir):
        super().__init__(scene_dir, resource_dir, data_dir)
        self.constant_variables = ['g', 'albedo', 'effAlbedo', 'ior']
        self.medium_param_generator = FixedMediumGenerator()
        self.searchlight_configuration = True


class ScatterDataFixedMediumSearchlightDeg2(ScatterDataFixedMediumSearchlight):
    def __init__(self, scene_dir, resource_dir, data_dir):
        super().__init__(scene_dir, resource_dir, data_dir)
        self.poly_order = 2


def get_subclasses(c):
    subclasses = c.__subclasses__()
    for sub in subclasses:
        subclasses += get_subclasses(sub)
    return subclasses


def get_config(config_name):
    # list all classes in the current module: these are all possible configs
    classes = get_subclasses(ScatterData)
    classes.append(ScatterData)
    string_names = [c.__name__.lower() for c in classes]
    class_name = config_name.lower()
    if class_name in string_names:
        return classes[string_names.index(class_name)]
    else:
        print('Data set config {} not found!'.format(config_name))
        quit()


def get_all_configs():
    # list all classes in the current module: these are all possible configs
    classes = ScatterData.__subclasses__()
    classes.append(ScatterData)
    return classes


def get_config_from_metadata(dataset_name, output_dir):
    metadata_file = os.path.join(output_dir, DATADIR, dataset_name, 'metadata.json')
    with open(metadata_file, 'r') as f:
        meta = json.load(f)
        return vae.datapipeline.get_config(meta['config'])


def get_mesh_generator(gen_name):
    # list all classes in the current module: these are all possible configs
    classes = MeshGenerator.__subclasses__()
    classes.append(MeshGenerator)
    string_names = [c.__name__.lower() for c in classes]
    class_name = gen_name.lower()
    if class_name in string_names:
        return classes[string_names.index(class_name)]
    else:
        print('Mesh set config {} not found!'.format(gen_name))
        quit()


def get_all_mesh_generators(scene_dir, resource_dir):
    # list all classes in the current module: these are all possible configs
    classes = MeshGenerator.__subclasses__()
    classes.append(MeshGenerator)
    return [c(scene_dir, resource_dir) for c in classes]
