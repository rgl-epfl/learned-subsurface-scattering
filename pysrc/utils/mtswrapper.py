import numpy as np
import os

import mitsuba
from mitsuba.core import *
import mitsuba.render
from mitsuba.render import SceneHandler


from mitsuba.render import PolyFitConfig
from mitsuba.render import PolyFitRecord
from mitsuba.render import MediumParameters


import utils.math
import utils.transforms

import vae.utils


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


def set_obj_from_dict(obj, d):
    for k in d.keys():
        if hasattr(obj, k):
            obj.__setattr__(k, d[k])


def poly_fit_config_to_dict(cfg):
    d = dict()
    d["regularization"] = cfg.regularization
    d["useSvd"] = cfg.useSvd
    d["useLightspace"] = cfg.useLightspace
    d["order"] = cfg.order
    d["hardSurfaceConstraint"] = cfg.hardSurfaceConstraint
    d["globalConstraintWeight"] = cfg.globalConstraintWeight
    d["kdTreeThreshold"] = cfg.kdTreeThreshold
    d["extractNormalHistogram"] = cfg.extractNormalHistogram
    d["useSimilarityKernel"] = cfg.useSimilarityKernel
    return d


def fitPolynomial(constraint_kd_tree, ref_pos, ref_dir, sigma_t, g, albedo, options, normal=None):
    pfRec = mitsuba.render.PolyFitRecord()
    pfRec.p = mts_p(ref_pos)
    pfRec.d = mts_v(ref_dir)
    if normal is not None:
        pfRec.n = mts_v(normal)

    pfRec.kernelEps = vae.utils.kernel_epsilon(g, sigma_t, albedo)
    set_obj_from_dict(pfRec.config, options)
    coeffs, pos_constraints, nor_constraints = mitsuba.render.Volpath3D.fitPolynomials(pfRec, constraint_kd_tree)
    return np.array(coeffs), mts_to_np(pos_constraints), mts_to_np(nor_constraints)


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


def rotate_polynomial(coeffs, ref_dir, poly_order):
    s, t = utils.math.onb_duff(ref_dir)
    n = mts_v(ref_dir)
    s = mts_v(s)
    t = mts_v(t)
    coeffs = mitsuba.render.Volpath3D.rotatePolynomial([float(c) for c in coeffs], s, t, n, poly_order)
    return np.array(coeffs)


def rotate_polynomial_azimuth(coeffs, ref_dir, normal, poly_order, inverse=False):
    # transf = mitsuba.render.Volpath3D.azimuthSpaceTransform(
    #     utils.mtswrapper.mts_v(ref_dir), utils.mtswrapper.mts_v(normal))
    # if inverse:
    #     full_transform = np.array(transf)
    # else:
    #     full_transform = np.array(transf).T

    transf = utils.transforms.to_azimuth_space(ref_dir, normal)
    transf = transf.T

    if inverse:
        transf = transf.T

    s = utils.mtswrapper.mts_v(transf[:, 0])
    t = utils.mtswrapper.mts_v(transf[:, 1])
    n = utils.mtswrapper.mts_v(transf[:, 2])

    coeffs = mitsuba.render.Volpath3D.rotatePolynomial([float(c) for c in coeffs], s, t, n, poly_order)
    return np.array(coeffs)


def adjust_ray_direction_for_polynomial(coeffs, ray_origin, ray_dir, geo_normal, poly_scale_factor):
    ray_origin_mts = mts_p(ray_origin)
    ray_dir_mts = mts_v(ray_dir)
    l = mitsuba.render.Volpath3D.adjustDirForPolynomialTracing(ray_origin_mts, ray_dir_mts,
                                                              [float(c) for c in coeffs], 
                                                              ray_origin_mts, poly_scale_factor, mts_v(geo_normal))

    # This should always be positive now: adjusted ray direction should 
    print(f"utils.math.dot(mts_to_np(l[0]), mts_to_np(l[1])): {utils.math.dot(mts_to_np(l[0]), mts_to_np(l[1]))}")
    return mts_to_np(l[0]), mts_to_np(l[1])


def azimuth_transform(ref_dir, normal):
    transf = mitsuba.render.Volpath3D.azimuthSpaceTransform(
        utils.mtswrapper.mts_v(ref_dir), utils.mtswrapper.mts_v(normal))
    return np.array(transf)



def get_default(d, k, default):
    if k in d:
        return d[k]
    else:
        return default


def sample_scattering_mesh(mesh_file, n_samples, batch_size, n_abs_samples, media, seed, kdtree, extra_options):
    assert (mesh_file)
    resource_dir = './resources'
    fileResolver = Thread.getThread().getFileResolver()
    fileResolver.appendPath(resource_dir)
    fileResolver.appendPath(os.path.split(mesh_file)[0])
    paramMap = StringMap()
    paramMap['meshfile'] = mesh_file
    scene = SceneHandler.loadScene(fileResolver.resolve(resource_dir + '/scene_shape_template.xml'), paramMap)
    scene.initialize()
    shape = scene.getShapes()[0]
    ignore_zero_scatter = get_default(extra_options, 'ignore_zero_scatter', True)
    fixed_in_dir = get_default(extra_options, 'fixed_in_dir', False)

    poly_fit_cfg = PolyFitConfig()
    set_obj_from_dict(poly_fit_cfg, extra_options)
    poly_fit_cfg.useLightspace = False
    importanceSamplePolys = get_default(extra_options, 'importance_sample_polys', False)

    if kdtree is not None:
        tmp_result = mitsuba.render.Volpath3D.samplePoly(shape, media, n_samples, batch_size, n_abs_samples, ignore_zero_scatter, False,
                                                         importanceSamplePolys, kdtree, fixed_in_dir, seed, poly_fit_cfg)
    else:
        compute_sh_coefficients = get_default(extra_options, 'compute_sh_coefficients', False)
        if compute_sh_coefficients:
            tmp_result = mitsuba.render.Volpath3D.sampleShCoeffs(
                scene, shape, media, n_samples, batch_size, n_abs_samples, ignore_zero_scatter, False, seed)
        else:
            tmp_result = mitsuba.render.Volpath3D.sample(
                scene, shape, media, n_samples, batch_size, n_abs_samples, ignore_zero_scatter, False, seed)
    return tmp_result


def sample_scattering_poly(n_samples, batch_size, n_abs_samples, media, poly_coeffs, seed, extra_options):
    ref_pos_mts = Point3(0, -0.0001, 0)
    ref_dir_mts = Vector3(0, -1, 0)

    if False:
        v = utils.math.sampleCosineDistribution()
        a = v[1]
        b = v[2]
        v[2] = a
        v[1] = b
        v = -v
        ref_dir_mts = utils.mtswrapper.mts_v(v)
        print(f"v: {v}")

    coeffs_list = [float(c) for c in poly_coeffs]
    ignore_zero_scatter = get_default(extra_options, 'ignore_zero_scatter', True)
    scale_factor = float(vae.utils.get_poly_scale_factor(
        vae.utils.kernel_epsilon(media[0].g, media[0].sigmaT[0], media[0].albedo[0])))
    tmp_result = mitsuba.render.Volpath3D.samplePolyFixedStart(coeffs_list, ref_pos_mts, ref_dir_mts, False,
                                                               scale_factor, media, n_samples, batch_size, n_abs_samples,
                                                               ref_pos_mts, ref_dir_mts, ignore_zero_scatter, False, seed)
    tmp_result['shapeCoeffs'] = [coeffs_list for p in tmp_result['inPos']]
    tmp_result['ior'] = [1.0 for p in tmp_result['inPos']]
    return tmp_result


def sample_scattering_poly2(ref_pos, ref_dir, n_samples, batch_size, n_abs_samples, media, poly_coeffs, seed, extra_options):
    ref_pos_mts = mts_p(ref_pos)
    ref_dir_mts = mts_v(ref_dir)
    coeffs_list = [float(c) for c in poly_coeffs]
    ignore_zero_scatter = get_default(extra_options, 'ignore_zero_scatter', True)
    scale_factor = float(vae.utils.get_poly_scale_factor(
        vae.utils.kernel_epsilon(media[0].g, media[0].sigmaT[0], media[0].albedo[0])))
    tmp_result = mitsuba.render.Volpath3D.samplePolyFixedStart(coeffs_list, ref_pos_mts, ref_dir_mts, False,
                                                               scale_factor, media, n_samples, batch_size, n_abs_samples,
                                                               ref_pos_mts, ref_dir_mts, ignore_zero_scatter, False, seed)
    tmp_result['shapeCoeffs'] = [coeffs_list for p in tmp_result['inPos']]
    tmp_result['ior'] = [1.0 for p in tmp_result['inPos']]
    return tmp_result
