import glob
import pickle
import re
import subprocess
import time

import numpy as np
import scipy.special
import skimage.transform
import tqdm
from PIL import Image, ImageDraw
from sklearn.preprocessing import PolynomialFeatures

import mitsuba
import mitsuba.core
import mitsuba.render
from utils.printing import *
from utils.math import onb_duff
import utils.math
import utils.transforms

from scipy.interpolate import RegularGridInterpolator

from vae.global_config import *


def covariance(X):
    """Estimates the covariance matrix of data X"""
    return X.T.dot(X) / X.shape[0]


def pca_transform(X):
    """Compute a whitening PCA transform of X"""
    cov = covariance(X)
    U, s, _ = np.linalg.svd(cov)
    n_dims = s.shape[0]
    sqrtS = np.diag(1 / np.sqrt(s[:n_dims]))
    W = U[:, :n_dims].dot(sqrtS)
    new_x = X.dot(W)
    return new_x


def discretize_polygon(polygon, voxelgrid_res):
    res = 4 * voxelgrid_res
    pixel_coords = ((polygon + 1) / 2 * res).astype(int)
    img = Image.new('L', (res, res), 0)
    pixel_coords = [(l[0], l[1]) for l in pixel_coords]
    ImageDraw.Draw(img).polygon(pixel_coords, outline=1, fill=1)
    # mask = np.flipud(np.array(img))
    mask = np.array(img)
    mask = skimage.transform.resize(mask, [voxelgrid_res, voxelgrid_res], preserve_range=True, mode='constant')
    voxel_grid = np.round(mask)
    return voxel_grid


def rescale_polygon(poly):
    return (poly - np.min(poly, 0)) / (np.max(poly, 0) - np.min(poly, 0)) * 1.8 - 0.9


def load_polygons(scene_dir):
    poly_files = sorted(glob.glob(scene_dir + '/poly*.pickle'))
    polygons = []
    for poly in poly_files:
        with open(poly, 'rb') as f:
            polygons.append(rescale_polygon(pickle.load(f)))
    return polygons


def implicit_function_cpp(poly, query_positions, eps_val=0.02):
    pos_mts = [mitsuba.core.Point2(r[0], r[1]) for r in poly]
    query_mts = [mitsuba.core.Point2(r[0], r[1]) for r in query_positions]
    res = mitsuba.render.Volpath2D.evalImplicitFunction(pos_mts, query_mts, eps_val)
    return {'f': res['val'],
            'dx': res['dx'],
            'dy': res['dy']}


def project_points_on_poly(polygon, points, get_normal=False):
    """Projects given points onto the closest point on the polygon"""

    min_dist = np.ones((points.shape[0], 1)) * 1000
    min_idx = np.zeros((points.shape[0], 1), dtype=np.int32)
    min_proj = np.zeros((points.shape[0], 2))
    min_normal = np.zeros((points.shape[0], 2))
    for i in range(polygon.shape[0]):
        p0 = polygon[i]
        p1 = polygon[(i + 1) % polygon.shape[0]]

        d = p1 - p0
        edge_len = np.sqrt(np.sum(d ** 2))
        d = d / edge_len
        d = d[np.newaxis, :]
        proj = np.sum((-p0 + points) * d, axis=1, keepdims=True)
        proj = np.maximum(0, np.minimum(proj, edge_len))
        proj = proj * d + p0
        dist = np.sqrt(np.sum((proj - points) ** 2, axis=1, keepdims=True))

        normal = -np.stack([-d[:, 1], d[:, 0]], axis=1)
        new_min_idx = dist < min_dist
        min_idx[new_min_idx[:, 0], :] = i
        min_dist[new_min_idx[:, 0], :] = dist[new_min_idx[:, 0], :]
        min_proj[new_min_idx[:, 0], :] = proj[new_min_idx[:, 0], :]
        min_normal[new_min_idx[:, 0], :] = normal

    if get_normal:
        return min_proj, min_normal
    else:
        return min_proj


def project_points_on_mesh(scene, points, ref_point, ref_dir, poly_coeffs, poly_order, use_local_coords, scale_factor, soft_projection=False):
    """Projects points onto scene geometry by tracing rays along the gradient (~surface normal) of the polynomial surface approximation"""
    if soft_projection:
        rate = 25.0
        rate = 1.0
        outPos = np.copy(points)
        grad = eval_poly_gradient(outPos, ref_point, ref_dir, poly_coeffs, use_local_coords, scale_factor)

        d = utils.math.normalize(grad)
        t = 0.0
        for i in range(3):
            # -- Standard gradient descent --
            # val = eval_poly(outPos, ref_point, ref_dir, poly_coeffs, use_local_coords, scale_factor)
            # grad = eval_poly_gradient(outPos, ref_point, ref_dir, poly_coeffs, use_local_coords, scale_factor)
            # outPos = outPos - rate * val * grad

            # -- Gradient descent along a line parametrized by t --
            # val = eval_poly(outPos, ref_point, ref_dir, poly_coeffs, use_local_coords, scale_factor)
            # grad = eval_poly_gradient(outPos, ref_point, ref_dir, poly_coeffs, use_local_coords, scale_factor)
            # t = t - rate * utils.math.dot(grad, d) * val
            # outPos = points + t * d

            # -- Newton method a long a line parametrized by t --
            val = eval_poly(outPos, ref_point, ref_dir, poly_coeffs, use_local_coords, scale_factor)
            grad = eval_poly_gradient(outPos, ref_point, ref_dir, poly_coeffs, use_local_coords, scale_factor)
            t = t - rate * val / (utils.math.dot(grad, d) * scale_factor)
            outPos = points + t * d

        outNormal = utils.math.normalize(grad)
        valid_pos = np.ones(outNormal.shape[0], dtype=np.bool)
        return outPos, outNormal, valid_pos
    else:
        gen_samples_mts = [mitsuba.core.Point(float(p[0]), float(p[1]), float(p[2])) for p in points]
        its_loc_mts = mitsuba.core.Point(float(ref_point[0]), float(ref_point[1]), float(ref_point[2]))
        dir_mts = mitsuba.core.Vector(float(ref_dir[0]), float(ref_dir[1]), float(ref_dir[2]))
        projected_points_mts, normal_mts, valid = mitsuba.render.Volpath3D.projectToSurface(scene, its_loc_mts, dir_mts, gen_samples_mts,
                                                                                            [float(c) for c in poly_coeffs], poly_order, use_local_coords, scale_factor)
        return np.array([[p.x, p.y, p.z] for p in projected_points_mts]), np.array([[p.x, p.y, p.z] for p in normal_mts]), valid


def get_legendre_transform(order):
    """Get a transform matrix which takes polynomials from the standard basis to the Legendre basis"""
    import sympy
    from sympy.abc import x, y, z

    if order > 3:
        raise ValueError('Supports only up to order 3 polynomials!')

    def legendre(v, order):
        return [1, v, sympy.Rational(1, 2) *(3 * v * v - 1), sympy.Rational(1, 2) * (5 * v * v * v - 3 * v)][:(order+1)]

    def digits_to_number(digits):
        return int(''.join([str(d) for d in digits]))

    def sort_key(poly, order):
        deg = sympy.degree(poly, gen=x) + sympy.degree(poly, gen=y) + sympy.degree(poly, gen=z)
        all_degs = [order - deg,  sympy.degree(poly, gen=x), sympy.degree(poly, gen=y), sympy.degree(poly, gen=z)]
        return digits_to_number(all_degs)


    leg_x = legendre(x, order)
    leg_y = legendre(y, order)
    leg_z = legendre(z, order)

    orig_basis = []
    for d in range(order + 1):
        for i in range(d + 1):
            for j in range(i + 1):
                dx = d - i
                dy = d - dx - j
                dz = d - dx - dy
                term = x ** dx * y ** dy * z ** dz
                orig_basis.append(term)

    new_basis = []
    for i in leg_x:
        for j in leg_y:
            for k in leg_z:
                new_p = i * j * k
                deg = sympy.degree(new_p, gen=x) + sympy.degree(new_p, gen=y) + sympy.degree(new_p, gen=z)
                if deg <= order:
                    new_basis.append(new_p)
    new_basis = sorted(new_basis, key=lambda poly: sort_key(poly, order))
    new_basis.reverse()
    for i, term in enumerate(new_basis):
        coeff = sympy.integrate(term * term, (x, -1, 1))
        coeff = sympy.integrate(coeff, (y, -1, 1))
        coeff = sympy.integrate(coeff, (z, -1, 1))
        new_basis[i] = term / sympy.sqrt(coeff)

    # For each term in the original basis: Integrate it against all new basis vectors to get coefficients
    transform_matrix = np.zeros((len(orig_basis), len(orig_basis)))
    for i, term in enumerate(orig_basis):
        for j, term2 in enumerate(new_basis):
            coeff = sympy.integrate(term * term2, (x, -1, 1))
            coeff = sympy.integrate(coeff, (y, -1, 1))
            coeff = sympy.integrate(coeff, (z, -1, 1))
            transform_matrix[j, i] = coeff
    return transform_matrix


def power_to_index(power, order):
    dim = power.shape[1]
    poly = PolynomialFeatures(order)
    poly.fit_transform(np.random.rand(1, dim))
    ref_powers = poly.powers_

    num_terms = np.max(ref_powers) + 1

    # Compute indices of the original terms in the polynomial
    flattened_powers = np.ravel_multi_index([ref_powers[:, i] for i in range(ref_powers.shape[1])],
                                            (num_terms, ) * dim)

    flat_power_to_power_idx = -np.ones(num_terms ** dim, dtype=np.int32)
    flat_power_to_power_idx[flattened_powers] = np.arange(0, ref_powers.shape[0], 1, dtype=np.int32)

    # Now compute indices of the query powers
    flat_query_power = np.ravel_multi_index([power[:, i] for i in range(power.shape[1])],
                                            (num_terms, ) * dim)

    return flat_power_to_power_idx[flat_query_power]


def deriv_matrix(dim, order, axis, a=False):
    poly = PolynomialFeatures(order)
    data = np.random.rand(1, dim)
    poly.fit_transform(data)
    powers = poly.powers_

    d_mat = np.zeros((powers.shape[0], powers.shape[0]))

    powers_deriv = np.array(powers)
    powers_deriv[:, axis] -= 1

    powers_deriv2 = np.array(powers_deriv)
    powers_deriv2[np.any(powers_deriv < 0, 1)] = 0
    indices = power_to_index(powers_deriv2, order)

    for i in range(powers.shape[0]):
        if powers_deriv[i, axis] < 0:
            continue
        d_mat[indices[i], i] = powers[i][axis]

    return d_mat


def basis_fun(pos_p, sampled_p, poly_order):
    return PolynomialFeatures(degree=poly_order).fit_transform(sampled_p - pos_p)


def kernel_epsilon(g, sigma_t, albedo, use_legacy_kernel=False):
    """Optimal kernel to fit polynomial to represent the surface for scattering"""
    # if use_legacy_kernel:
    #     return float(FIT_KERNEL_EPSILON / (np.maximum((1 - g), 0.2) * sigma_t ** 2))




    if type(g) is np.ndarray and g.size >= 1:
        albedo = np.mean(albedo, -1)
        sigma_t = np.mean(sigma_t, -1)
        return np.array(mitsuba.render.Volpath3D.kernelEpsilon(g.ravel().tolist(), sigma_t.tolist(), albedo.tolist()))[:, np.newaxis]

    if use_legacy_kernel:
        return mitsuba.render.Volpath3D.kernelEpsilonLegacy([float(g)], [float(sigma_t)], [float(albedo)])[0]
    else:
        return mitsuba.render.Volpath3D.kernelEpsilon([float(g)], [float(sigma_t)], [float(albedo)])[0]


def get_poly_scale_factor(kernel_eps):
    return 1.0 / np.sqrt(kernel_eps)



def polynomial_to_voxel_grid(current_pos, current_normal, poly_coeffs, poly_order,
                             tangent_space, extent=[-1, 1, -1, 1], res=64):
    # Evaluate taylor expansion
    dim = current_pos.shape[1]
    coords = []
    for d in range(dim):
        coords.append(np.linspace(extent[2 * d], extent[2 * d + 1], res) + 0.5 / res)
    coords_tuple = tuple(coords)
    if dim == 2:  # TODO: Also use ij indexing for 2D
        components = np.meshgrid(*coords, indexing='xy')
    elif dim == 3:
        components = np.meshgrid(*coords, indexing='ij')
    coords = np.stack([c.ravel() for c in components], 1)

    if tangent_space:  # Convert to tangent space
        coords_ts = world_to_local_np(current_pos, current_normal, coords, True)
        b = basis_fun(current_pos * 0.0, coords_ts, poly_order)
    else:
        b = basis_fun(current_pos, coords, poly_order)

    f_taylor = np.reshape(b.dot(poly_coeffs), [res] * dim)
    min_pos = np.array([[extent[2 * d] for d in range(dim)]])
    max_pos = np.array([[extent[2 * d + 1] for d in range(dim)]])
    bb_diag = max_pos - min_pos

    if dim == 2:
        fit_level = (res * (current_pos - min_pos) / bb_diag).astype(np.int32).ravel()
        fit_level = f_taylor[fit_level[1], fit_level[0]]
    else:
        fit_level = RegularGridInterpolator(coords_tuple, f_taylor)(current_pos)
    f_taylor -= fit_level
    return f_taylor


def polynomial_to_voxel_grid_new(pos, in_dir, normal, poly_coeffs, poly_order,
                                 prediction_space, extent=[-1, 1, -1, 1], res=64):
    # Evaluate taylor expansion
    dim = pos.shape[1]
    coords = []
    for d in range(dim):
        coords.append(np.linspace(extent[2 * d], extent[2 * d + 1], res) + 0.5 / res)
    components = np.meshgrid(*coords, indexing='ij')
    coords = np.stack([c.ravel() for c in components], 1)

    coords_transformed, transf = utils.transforms.transform_to(coords, pos, in_dir, normal, prediction_space)
    b = basis_fun(pos * 0.0, coords_transformed, poly_order)
    return np.reshape(b.dot(poly_coeffs), [res] * dim)


def eval_poly(points, ref_pos, ref_dir, poly_coeffs, tangent_space=False, scale=1.0):
    n_coeffs = poly_coeffs.size
    poly_order = extract_poly_order_from_n_coeffs(n_coeffs)
    if tangent_space:
        points = world_to_local_np(ref_pos, ref_dir, points, True)
    else:
        points = points - ref_pos
    points *= scale
    points = np.atleast_2d(points)
    return np.sum(PolynomialFeatures(poly_order).fit_transform(points) * poly_coeffs, axis=1, keepdims=True)


def eval_poly_gradient(query_pos, ref_pos, ref_dir, poly_coeffs, tangent_space, scale=1.0):
    query_pos = np.atleast_2d(query_pos)
    if tangent_space:
        pos = world_to_local_np(ref_pos, ref_dir, query_pos, True)
    else:
        pos = query_pos - ref_pos

    pos *= scale

    dim = pos.shape[1]
    pos = np.atleast_2d(pos)
    poly_coeffs = np.atleast_2d(poly_coeffs)
    poly_order = extract_poly_order_from_n_coeffs(poly_coeffs.shape[1])
    deriv = np.zeros_like(pos)
    for d in range(dim):
        deriv_mat = deriv_matrix(dim, poly_order, d)
        # Derive the coeffs
        deriv_coeffs = (deriv_mat @ poly_coeffs.T).T
        deriv_d = np.sum(PolynomialFeatures(poly_order).fit_transform(pos) * deriv_coeffs, axis=1)
        deriv[:, d] = deriv_d

    # Rotate deriv to be in world space again
    if tangent_space:
        t1, t2 = onb_duff(ref_dir)
        deriv = t1 * deriv[:, 0][:, None] + t2 * deriv[:, 1][:, None] + ref_dir * deriv[:, 2][:, None]
    return deriv


def inv_dist_kernel(pos_p, sampled_p, sigma2):
    return 1.0 / (np.sum((pos_p - sampled_p) ** 2, axis=1) + sigma2)


def gauss_kernel(pos_p, sampled_p, sigma2):
    w = np.exp(-np.sum((pos_p - sampled_p) ** 2, axis=1) / sigma2)
    w[np.sum((pos_p - sampled_p) ** 2, axis=1) > 4 * sigma2] = 0.0
    return w


def implicit_function_polynomial(pos_p, sampled_p, sampled_n, poly_order, sigma2=0.001, kernel='invdist2', eval_coords=None):
    pos_p = np.atleast_2d(pos_p)

    if kernel == 'invdist2':
        kernel_fun = inv_dist_kernel
    elif kernel == 'gaussian':
        kernel_fun = gauss_kernel

    weights = kernel_fun(pos_p, sampled_p, sigma2)

    if eval_coords is not None:
        weights_coords = kernel_fun(pos_p, eval_coords, sigma2)

    basis_function = basis_fun(pos_p, sampled_p, poly_order)

    A = np.sqrt(weights)[:, np.newaxis] * basis_function
    b = np.sqrt(weights)[:, np.newaxis] * np.sum(sampled_n * (pos_p - sampled_p), axis=1, keepdims=True)
    c, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

    value = basis_fun(pos_p, pos_p, poly_order).dot(c).ravel()

    if eval_coords is not None:
        return value, c, weights_coords
    else:
        return value, c


def albedo_to_effective_albedo(albedo):
    return -np.log(1.0 - albedo * (1.0 - np.exp(-8.0))) / 8.0

def get_alphap(albedo, g, sigmat):
    sigmas = albedo * sigmat
    sigmaa = sigmat - sigmas
    return (1 - g) * sigmas / ((1 - g) * sigmas + sigmaa)


def get_sigmatp(albedo, g, sigmat):
    sigmas = albedo * sigmat
    sigmaa = sigmat - sigmas
    return (1 - g) * sigmas + sigmaa


def effective_albedo_to_albedo(effective_albedo):
    return (1 - np.exp(-8 * effective_albedo)) / (1 - np.exp(-8))


def implicit_function_gradient_constraint(pos_p, sampled_p, sampled_n, poly_order, sigma2=0.001, kernel='gaussian',
                                          eval_coords=None, reg=FIT_REGULARIZATION, gradweight=1.0, rescaling=1.0):
    pos_p = np.atleast_2d(pos_p)

    if kernel == 'invdist2':
        kernel_fun = inv_dist_kernel
    elif kernel == 'gaussian':
        kernel_fun = gauss_kernel
    else:
        raise ValueError('Unknown kernel function {}'.format(kernel))

    weights = kernel_fun(pos_p, sampled_p, sigma2)
    if eval_coords is not None:
        weights_coords = kernel_fun(pos_p, eval_coords, sigma2)

    pos_p = pos_p * rescaling
    sampled_p = sampled_p * rescaling
    basis_function = basis_fun(pos_p, sampled_p, poly_order)

    dim = pos_p.shape[1]

    constraint_mats = []
    b_values = []
    A_value = np.sqrt(weights)[:, np.newaxis] * basis_function
    constraint_mats.append(A_value)
    b_values.append(np.sqrt(weights)[:, np.newaxis] * np.zeros((pos_p.shape[0], 1)))

    for d in range(dim):
        deriv = deriv_matrix(dim, poly_order, d)
        A_deriv = gradweight * np.sqrt(weights)[:, np.newaxis] * basis_function.dot(deriv)
        constraint_mats.append(A_deriv)
        b_values.append(gradweight * np.sqrt(weights)[:, np.newaxis] * sampled_n[:, d][:, np.newaxis])

    A = np.concatenate(constraint_mats, axis=0)
    b = np.concatenate(b_values, axis=0)
    c, _, _, _ = np.linalg.lstsq(A.T.dot(A) + reg * np.eye(A.shape[1]), A.T.dot(b), rcond=None)

    print('Error {}'.format(np.mean((A.dot(c) - b) ** 2)))

    value = basis_fun(pos_p, pos_p, poly_order).dot(c).ravel()

    if eval_coords is not None:
        return value, c, weights_coords, weights
    else:
        return value, c, weights


def implicit_function_polynomial2(pos_p, sampled_p, sampled_v, poly_order, sigma2=0.001, kernel='invdist2', eval_coords=None, weight_clip_threshold=0):
    """Fits polynomial to a locally prescribed list of values"""
    pos_p = np.atleast_2d(pos_p)

    if kernel == 'invdist2':
        kernel_fun = inv_dist_kernel
    elif kernel == 'gaussian':
        kernel_fun = gauss_kernel

    weights = kernel_fun(pos_p, sampled_p, sigma2)
    weights[weights < weight_clip_threshold] = 0

    if eval_coords is not None:
        weights_coords = kernel_fun(pos_p, eval_coords, sigma2)
        weights_coords[weights_coords < weight_clip_threshold] = 0

    basis_function = basis_fun(pos_p, sampled_p, poly_order)

    A = np.sqrt(weights)[:, np.newaxis] * basis_function
    b = np.sqrt(weights)[:, np.newaxis] * sampled_v[:, np.newaxis]
    c, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

    value = basis_fun(pos_p, pos_p, poly_order).dot(c).ravel()

    if eval_coords is not None:
        return value, c, weights_coords
    else:
        return value, c


def implicit_function_iterated(poly, pos_p, poly_order, sigma2=0.001, kernel='invdist2', weight_clip_threshold=0, use_tangent_space=True, pos_n=None):
    """First constructs an implicit function on the grid and then fits a local polynomial"""
    x, y = np.meshgrid(np.linspace(-1, 1, 64), np.linspace(-1, 1, 64))
    query_positions = np.stack([x.ravel(), y.ravel()], axis=1)
    impl_f = implicit_function(poly, query_positions, 0.01, True)
    res_values = np.reshape(impl_f['f'], x.shape)

    if use_tangent_space:
        # Transform query positions to tangent space
        query_positions = to_tangent_space(query_positions, pos_n, pos_p)
        pos_p = 0.0 * pos_p

    value, coeffs = implicit_function_polynomial2(pos_p, query_positions, res_values.ravel(),
                                                  poly_order, sigma2, kernel, weight_clip_threshold=weight_clip_threshold)
    return value, coeffs


def get_shape_feature_normalization(traindata_file, feature_name):
    with open(traindata_file + '_stats.pickle', 'rb') as f:
        stats = pickle.load(f)
        shapeCoeffs_mean = stats['{}_mean'.format(feature_name)]
        shapeCoeffs_var = 1.0 / (stats['{}_stdinv'.format(feature_name)]) ** 2
    return shapeCoeffs_mean, shapeCoeffs_var


def get_feature_stats(traindata_file):
    with open(traindata_file + '_stats.pickle', 'rb') as f:
        stats = pickle.load(f)
        return stats


def world_to_local_np(in_pos, in_normal, out_pos_ws, predict_in_tangent_space):
    in_pos = np.atleast_2d(in_pos)
    in_normal = np.atleast_2d(in_normal)
    if in_pos.shape[-1] == 3:
        if predict_in_tangent_space:
            rel_out_pos = out_pos_ws - in_pos
            in_normal = in_normal / np.sqrt(np.sum(in_normal ** 2, axis=-1, keepdims=True))
            tangent1, tangent2 = onb_duff(in_normal)
            c_n = np.sum(rel_out_pos * in_normal, axis=-1)
            c_t1 = np.sum(rel_out_pos * tangent1, axis=-1)
            c_t2 = np.sum(rel_out_pos * tangent2, axis=-1)
            rel_out_pos = np.stack([c_t1, c_t2, c_n], axis=-1)
        else:
            rel_out_pos = out_pos_ws - in_pos
    else:
        if predict_in_tangent_space:
            rel_out_pos = out_pos_ws - in_pos
            tangent = np.stack([-in_normal[..., 1], in_normal[..., 0]], axis=-1)
            c_n = np.sum(rel_out_pos * in_normal, axis=-1)
            c_t = np.sum(rel_out_pos * tangent, axis=-1)
            rel_out_pos = np.stack([c_n, c_t], axis=-1)
        else:
            rel_out_pos = out_pos_ws - in_pos
    return rel_out_pos


def world_dir_to_local_np(direction, normal):
    '''Transforms a world space vector into tangent space'''
    normal = normal / np.sqrt(np.sum(normal ** 2, axis=-1, keepdims=True))
    direction = direction / np.sqrt(np.sum(direction ** 2, axis=-1, keepdims=True))
    tangent1, tangent2 = onb_duff(normal)
    c_n = np.sum(direction * normal, axis=-1)
    c_t1 = np.sum(direction * tangent1, axis=-1)
    c_t2 = np.sum(direction * tangent2, axis=-1)
    return np.stack([c_t1, c_t2, c_n], axis=-1)


def to_ref_dir_coords(x, ref_dir):
    tangent1, tangent2 = onb_duff(ref_dir)
    c_t1 = np.sum(x * tangent1, axis=-1)
    c_t2 = np.sum(x * tangent2, axis=-1)
    c_n = np.sum(x * ref_dir, axis=-1)
    return np.stack([c_t1, c_t2, c_n], axis=-1)

def world_to_local_np_new(x, in_pos, in_dir, in_normal, local_space='TS'):
    in_pos = np.atleast_2d(in_pos)
    in_normal = np.atleast_2d(in_normal)
    x = np.atleast_2d(x)
    rel_out_pos = x - in_pos
    if local_space == 'WS':
        return rel_out_pos
    elif local_space == 'LS':
        return to_ref_dir_coords(rel_out_pos, -in_dir)
    elif local_space == 'AS':
        # TODO: This should support batch mode
        transf = utils.mtswrapper.azimuth_transform(-in_dir, in_normal)
        return (transf @ rel_out_pos[:, None])[..., 0]
    elif local_space == 'TS':
        return to_ref_dir_coords(rel_out_pos, in_normal)


def cartesian_to_spherical(direction):
    xy = direction[:, 0] ** 2 + direction[:, 1] ** 2
    coords = np.zeros_like(direction)
    coords[:, 0] = np.sqrt(xy + direction[:, 2] ** 2)
    coords[:, 1] = np.arctan2(np.sqrt(xy), direction[:, 2])  # for elevation angle defined from Z-axis down
    # coords[:,4] = np.arctan2(direction[:,2], np.sqrt(xy)) # for elevation angle defined from XY-plane up
    coords[:, 2] = np.arctan2(direction[:, 1], direction[:, 0])
    return coords


def square_to_disk_concentric(pos):
    """Code from http://psgraphics.blogspot.com/2011/01/improved-code-for-concentric-map.html (improved version with fix from comment)"""
    pos = np.atleast_2d(pos)
    pos = 2.0 * pos - 1.0
    quadrant_1_or_3 = np.abs(pos[:, 0]) < np.abs(pos[:, 1])
    r = np.copy(pos[:, 0])
    phi = (np.pi / 4.0) * (pos[:, 1] / pos[:, 0])
    r[quadrant_1_or_3] = np.copy(pos[quadrant_1_or_3, 1])
    phi[quadrant_1_or_3] = (np.pi / 2.0) - (np.pi / 4.0) * (pos[quadrant_1_or_3, 0] / pos[quadrant_1_or_3, 1])
    phi[(pos[:, 0] == 0) * (pos[:, 1] == 0)] = 0.0
    return np.stack([r * np.cos(phi), r * np.sin(phi)], axis=1)


def disk_to_square_concentric(pos):
    """ Inverse square to disk, adapted from Mitsuba 2"""
    pos = np.atleast_2d(pos)
    quadrant_0_or_2 = np.abs(pos[:, 0]) > np.abs(pos[:, 1])
    r_sign = np.copy(pos[:, 1])
    r_sign[quadrant_0_or_2] = pos[quadrant_0_or_2, 0]
    r = np.copysign(np.sqrt(np.sum(pos ** 2, axis=1)), r_sign)
    phi = np.arctan2(np.sign(r_sign) * pos[:, 1], np.sign(r_sign) * pos[:, 0])
    t = phi * 4 / np.pi
    t[np.logical_not(quadrant_0_or_2)] = 2 - t[np.logical_not(quadrant_0_or_2)]
    t *= r
    a = np.copy(t)
    b = np.copy(r)
    a[quadrant_0_or_2] = r[quadrant_0_or_2]
    b[quadrant_0_or_2] = t[quadrant_0_or_2]
    return np.stack([(a + 1) * 0.5, (b + 1) * 0.5], axis=1)


def hemisphere_to_square(d):
    """Transforms vector [t_1, t_2, n] to a point in a [0, 1] x [0, 1] square by projecting and transforming"""
    return disk_to_square_concentric(d[:, :2])


def shape_feat_name_to_num_coeff(feat_name, dim):
    if feat_name == 'shCoeffs':
        return 8

    if not feat_name:
        return None
    m = re.search(r'\d+$', feat_name)
    if m:
        n = int(m.group())
        return int(scipy.special.comb(dim + n, n))
    return None


def extract_poly_order_from_feat_name(feat_name):
    m = re.search(r'\d+$', feat_name)
    if m:
        n = int(m.group())
        return n
    return None


def extract_poly_order_from_n_coeffs(n):
    m = {PolynomialFeatures(d).fit_transform(np.random.rand(1, 3)).shape[1]: d for d in range(20)}
    return m[n] if n in m else None


def sample_polygon(polygon, n_samples):
    edge_lens = np.sqrt(np.sum((polygon - np.roll(polygon, -1, axis=0)) ** 2, 1))
    edge_lens = edge_lens / np.sum(edge_lens)
    edge_idx_samples = np.random.choice(np.arange(0, polygon.shape[0], 1), n_samples, p=edge_lens)
    edge_t_samples = np.random.rand(n_samples, 1)

    p0 = polygon[edge_idx_samples, :]
    p1 = polygon[(edge_idx_samples + 1) % polygon.shape[0], :]

    sampled_p = p0 * (1 - edge_t_samples) + p1 * edge_t_samples
    d = p1 - p0
    d = d / np.sqrt(np.sum(d ** 2, 1, keepdims=True))
    sampled_n = -np.stack([-d[:, 1], d[:, 0]], 1)
    return sampled_p, sampled_n


def index_polygon(path_t, polygon):
    edge_lens = np.sqrt(np.sum((polygon - np.roll(polygon, -1, axis=0)) ** 2, 1))
    total_edge_len = np.sum(edge_lens)
    path_t = path_t * total_edge_len
    edge_lens_cdf = np.cumsum(edge_lens)

    current_len = path_t
    idx = np.argmax(edge_lens_cdf > current_len)
    if idx > 0:
        current_len -= edge_lens_cdf[idx - 1]
    edge_t = current_len / edge_lens[idx]

    p0 = polygon[idx]
    p1 = polygon[(idx + 1) % polygon.shape[0]]
    current_pos = (1 - edge_t) * p0 + edge_t * p1
    return current_pos, p0, p1


def rotate_polynomial(coeffs, d, dim, order):
    if dim != 2:
        raise ValueError('Currently only dim 2 supported')
    d = d / np.sqrt(np.sum(d ** 2))
    R = np.array([[d[0], -d[1]], [d[1], d[0]]])

    poly = PolynomialFeatures(order)
    poly.fit_transform(np.random.rand(1, dim))
    T = np.zeros((poly.powers_.shape[0], poly.powers_.shape[0]))
    for pi, power in enumerate(poly.powers_):
        l = power[0]
        m = power[1]
        for pj, tpower in enumerate(poly.powers_):
            a = tpower[0]
            b = tpower[1]
            if l - b + m == a:
                coeff = 0
                min_i = np.maximum(b - l, 0)
                max_i = np.minimum(b, m)
                for i in range(min_i, max_i + 1):
                    r_00 = R[0, 0] ** (l - b + i)
                    r_01 = R[0, 1] ** (b - i)
                    r_10 = R[1, 0] ** (m - i)
                    r_11 = R[1, 1] ** i
                    coeff += scipy.special.comb(l, b - i) * scipy.special.comb(m, i) * r_00 * r_01 * r_10 * r_11
                T[pj, pi] = coeff
    return T.dot(coeffs.T).T


def relative_angle_to_direction(angle, ref_dir):
    """Converts an angle relative to a reference direction to a direction vector (in 2D)"""
    ref_dir = ref_dir.ravel()
    v = np.array([np.cos(angle), np.sin(angle)])
    ref_dir = ref_dir / np.sqrt(np.sum(ref_dir ** 2))
    R = np.array([ref_dir, np.array([-ref_dir[1], ref_dir[0]])]).T
    return R.dot(v)


def mts_v(v):
    v = v.ravel()
    return mitsuba.core.Vector3(float(v[0]), float(v[1]), float(v[2]))


def mts_p(p):
    p = p.ravel()
    return mitsuba.core.Point3(float(p[0]), float(p[1]), float(p[2]))


def mts_to_np(data):
    if type(data) is list:
        return np.array([np.array([p[0], p[1], p[2]]) for p in data])
    else:
        return np.array([data[0], data[1], data[2]])


def medium_params_list(albedo, sigmaT, g, ior):
    m = []
    for i in range(len(albedo)):
        m.append(mitsuba.render.MediumParameters())
        m[-1].albedo = albedo[i]
        m[-1].g = g[i]
        m[-1].sigmaT = sigmaT[i]
        m[-1].ior = ior[i]
    return m


def add_variants(module_globals, base_classes, param_overwrite_dict, suffix):
    """Creates new configuration classes automatically by extending the provided base classes.
       The param_overwrite_dict is used to overwrite values and suffix
       is the suffix attached to the class name."""
    for c in base_classes:
        def new_init(self):
            super(self.__class__, self).__init__()
            for k in param_overwrite_dict:
                self.__dict__[k] = param_overwrite_dict[k]
        class_name = c.__name__ + suffix
        init_name = class_name + '__init'
        module_globals[init_name] = new_init
        generatedClass = type(class_name, (c,), {
            "__init__": module_globals[init_name],
        })
        module_globals[generatedClass.__name__] = generatedClass



def extract_render_time_from_log(filename):
    output = subprocess.check_output(f'exrheader {filename}', shell=True, encoding='UTF-8')
    output = output.split('\n')
    for line in output:
        if '[RenderJob] Render time:' in line:
            timing = float(line.split(' ')[-1][:-1])
            return timing

def extract_preprocess_time_from_log(filename):
    output = subprocess.check_output(f'exrheader {filename}', shell=True, encoding='UTF-8')
    output = output.split('\n')
    preproc_time = 0.0
    for line in output:
        if '[VaeScatter] Preprocessing time:' in line:
            preproc_time += float(line.split(' ')[-1][:-1])

    return preproc_time