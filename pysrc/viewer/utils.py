

import os
import pickle
import time
from enum import Enum
from multiprocessing import Process, Value

import numpy as np
from skimage import measure

import mitsuba
import mitsuba.render
import nanogui
import vae.config
import vae.datapipeline
import vae.utils
from train_scattering_model import OUTPUT3D
from mitsuba.core import *
from utils.experiments import load_config
from utils.gui import FilteredListPanel, LabeledSlider, add_checkbox
import utils.math
from utils.math import onb_duff
from utils.mesh import sample_mesh
from vae.global_config import (DATADIR, DATADIR3D, FIT_KDTREE_THRESHOLD,
                               FIT_REGULARIZATION,
                               POINTDENSITY, RESOURCEDIR, SCENEDIR3D)
from vae.model import generate_new_samples
from vae.visualization import visualize_reconstructed_samples_3d
from viewer.datasources import (FacetedTriangleMesh, PointCloud, TriangleMesh,
                                VectorCloud)
from viewer.render import get_view_ray_dir
from viewer.viewer import GLTexture, ViewerApp

import trimesh


class AngularParametrization(Enum):
    CONCENTRIC = 0
    PROJECTION = 1
    POLAR = 2
    WORLDSPACE = 3
    CONCENTRIC_RESCALED = 4


class ViewerState:
    def __init__(self, scatterviewer, inPos=None, coeffs=None, need_projection=False, ws_coeffs=False, poly_order=3, prediction_space='LS'):
        self.poly_order = poly_order
        self.prediction_space = prediction_space
        self.coeffs = coeffs
        self.ws_coeffs = ws_coeffs
        self.viewer = scatterviewer
        self.need_projection = need_projection
        if inPos is None:
            self.inPos = self.viewer.its_loc
        else:
            self.inPos = inPos

        self.hist, self.hist_mesh, self.rec_mesh = None, None, None
        self.out_dir_vectors, self.unproj_points, self.unproj_points_pc = None, None, None
        self.points_pc = None
        self.projection_vectors = None

        self.points = np.array([]).reshape((0, 3))
        self.point_colors = np.array([]).reshape((0, 3))
        self.point_unproj_colors = np.array([]).reshape((0, 3))
        self.unproj_points = np.array([]).reshape((0, 3))
        self.normals = np.array([]).reshape((0, 3))
        self.out_dirs = np.array([]).reshape((0, 3))
        self.n_bounces = np.array([], dtype=np.int64).reshape((0))

        kernel_eps = vae.utils.kernel_epsilon(self.viewer.g, self.viewer.sigma_t, self.viewer.albedo)
        self.poly_scale_factor = float(vae.utils.get_poly_scale_factor(kernel_eps))

    def set_out_dirs(self, out_dirs):
        self.out_dirs = out_dirs
        self.out_dir_vectors = VectorCloud(self.points, self.out_dirs, 1.0, np.array([1, 1, 0]))

    def append(self, points, unproj_points=None, normals=None, out_dirs=None, n_bounces=None, point_colors=None, point_unproj_colors=None):
        self.hist = None
        self.hist_mesh = None

        if self.need_projection and unproj_points is None:
            self.unproj_points = np.concatenate([self.unproj_points, points])
            soft_projection = False
            ref_dir = -self.viewer.inDirection if self.prediction_space == 'LS' else self.viewer.face_normal

            if self.prediction_space == 'AS' and False:
                coeffs = utils.mtswrapper.rotate_polynomial_azimuth(self.coeffs, -self.viewer.inDirection, self.viewer.face_normal, self.poly_order, True)
                outPos, outNormal, valid_pos = vae.utils.project_points_on_mesh(self.viewer.scene, points,
                                                                                self.inPos, ref_dir, coeffs,
                                                                                self.poly_order, False,
                                                                                self.poly_scale_factor, soft_projection)
            else:
                outPos, outNormal, valid_pos = vae.utils.project_points_on_mesh(self.viewer.scene, points,
                                                                                self.inPos, ref_dir, self.coeffs,
                                                                                self.poly_order, self.prediction_space != 'WS',
                                                                                self.poly_scale_factor, soft_projection)

            self.points = np.concatenate([self.points, outPos[valid_pos]])
            self.projection_vectors = VectorCloud(
                points[valid_pos], -points[valid_pos] + outPos[valid_pos], 1.0, np.array([1, 0, 1]))

            self.normals = np.concatenate([self.normals, outNormal[valid_pos]])

            if out_dirs is not None:
                self.out_dirs = np.concatenate([self.out_dirs, out_dirs[valid_pos]])
            if n_bounces is not None:
                self.n_bounces = np.concatenate([self.n_bounces, n_bounces[valid_pos]])
        else:
            self.points = np.concatenate([self.points, points])
            if out_dirs is not None:
                self.out_dirs = np.concatenate([self.out_dirs, out_dirs])
            if normals is not None:
                self.normals = np.concatenate([self.normals, normals])
            if n_bounces is not None:
                self.n_bounces = np.concatenate([self.n_bounces, n_bounces])
            if unproj_points is not None:
                self.unproj_points = np.concatenate([self.unproj_points, unproj_points])
            if point_colors is not None:
                self.point_colors = np.concatenate([self.point_colors, point_colors])

        if point_unproj_colors is not None:
            self.point_unproj_colors = np.concatenate([self.point_unproj_colors, point_unproj_colors])

        self.points_pc = PointCloud(self.points, self.point_colors)
        self.unproj_points_pc = PointCloud(self.unproj_points, self.point_unproj_colors)
        if self.out_dirs.shape[0] > 0:
            self.out_dir_vectors = VectorCloud(
                self.points, self.out_dirs, 1.0, np.array([1, 1, 0]))

    def get_mesh_histogram(self, scene, mesh, outPos, outNormal):
        if outNormal is None:
            return None

        triangle_values = np.array(scene.meshHistogram(
            [vae.utils.mts_p(p) for p in outPos], [vae.utils.mts_v(p) for p in outNormal]))
        return FacetedTriangleMesh(mesh.vertices, mesh.faces, triangle_values)

    def get_angular_histogram(self, query_point, query_radius, parametrization, eta, min_bounces, max_bounces):

        if self.out_dirs is None:
            print('Could not find outdirs')
            return

        cond = np.array([True] * self.points.shape[0])
        if self.n_bounces.shape[0] > 0:
            cond = cond & (self.n_bounces >= min_bounces)
            if max_bounces > 0:
                cond = cond & (self.n_bounces <= max_bounces)

        # Only use points inside of a certain query regions
        cond = cond & (np.sqrt(np.sum((self.points - query_point) ** 2, axis=1)) <= query_radius)
        out_dirs = self.out_dirs[cond, :]
        normals = self.normals[cond, :]
        out_dirs_ts = vae.utils.world_dir_to_local_np(out_dirs, normals)
        if parametrization == AngularParametrization.PROJECTION:
            uv = out_dirs_ts[:, :2]
            return np.histogram2d(uv[:, 0], uv[:, 1], 64, range=[[-1, 1], [-1, 1]])[0]
        elif parametrization == AngularParametrization.CONCENTRIC:
            uv = vae.utils.hemisphere_to_square(out_dirs_ts)
            return np.histogram2d(uv[:, 0], uv[:, 1], 64, range=[[0, 1], [0, 1]])[0]
        elif parametrization == AngularParametrization.CONCENTRIC_RESCALED:
            uv = ((2.0 * vae.utils.hemisphere_to_square(out_dirs_ts) - 1.0) * eta + 1.0) * 0.5
            return np.histogram2d(uv[:, 0], uv[:, 1], 64, range=[[0, 1], [0, 1]])[0]
        elif parametrization == AngularParametrization.POLAR:
            uv = vae.utils.cartesian_to_spherical(out_dirs_ts)[:, 1:]
            return np.histogram2d(uv[:, 0], uv[:, 1], 64, range=[[0, np.pi], [-np.pi, np.pi]])[0]
        elif parametrization == AngularParametrization.WORLDSPACE:
            uv = vae.utils.cartesian_to_spherical(out_dirs)[:, 1:]
            return np.histogram2d(uv[:, 0], uv[:, 1], 64, range=[[0, np.pi], [-np.pi, np.pi]])[0]

    def get_histogram(self):
        if self.hist is None:
            self.hist, _ = np.histogramdd(self.points, bins=64, range=[[self.viewer.min_pos[0], self.viewer.max_pos[0]], [
                self.viewer.min_pos[1], self.viewer.max_pos[1]], [self.viewer.min_pos[2], self.viewer.max_pos[2]]])
            self.hist = self.hist / np.max(self.hist)
        return self.hist

    def get_mesh_histogram(self):
        if self.hist_mesh is None:
            if self.normals is None:
                return None
            triangle_values = np.array(self.viewer.scene.meshHistogram(
                [vae.utils.mts_p(p) for p in self.points], [vae.utils.mts_v(p) for p in self.normals]))
            self.hist_mesh = FacetedTriangleMesh(self.viewer.mesh.mesh.vertices,
                                                 self.viewer.mesh.mesh.faces, triangle_values)

        return self.hist_mesh

    def get_reconstructed_mesh(self):
        res = 128
        v = self.viewer
        if self.rec_mesh is None and self.coeffs is not None:
            self.rec_mesh = get_coeff_trianglemesh(self.inPos, v.inDirection, v.face_normal, self.coeffs,
                                                   self.poly_order, self.poly_scale_factor,
                                                   v.min_pos, v.max_pos, res, self.prediction_space)

        return self.rec_mesh


def create_mts_scene_from_mesh(mesh_file, resource_dir='./resources'):
    """Creates and initializes Mitsuba scene from a given mesh file"""
    fileResolver = Thread.getThread().getFileResolver()
    fileResolver.appendPath(resource_dir)
    fileResolver.appendPath(os.path.split(mesh_file)[0])
    paramMap = StringMap()
    paramMap['meshfile'] = mesh_file
    scene = mitsuba.render.SceneHandler.loadScene(fileResolver.resolve(
        resource_dir + '/scene_shape_template.xml'), paramMap)
    scene.initialize()
    return scene


def dump_scattering_config(viewer, output_dir=os.path.join(vae.global_config.OUTPUT3D, 'figures', 'scatterpoints')):
    """Dumps everything which is needed to render the scattering from the viewer in high quality in mitsuba"""

    data = {}
    data['points'] = viewer.viewer_data.points
    data['unproj_points'] = viewer.viewer_data.unproj_points
    data['inPos'] = viewer.its_loc
    data['cameraPos'] = viewer.camera.pos
    data['inDir'] = viewer.inDirection
    data['mesh'] = viewer.mesh_file
    data['albedo'] = viewer.albedo
    data['g'] = viewer.g
    data['sigmat'] = viewer.sigma_t
    data['eta'] = viewer.eta
    data['coeffs'] = viewer.viewer_data.coeffs
    data['scale_factor'] = viewer.viewer_data.poly_scale_factor

    os.makedirs(output_dir, exist_ok=True)
    num = utils.namegen.get_next_experiment_number(output_dir, True)
    output_file = os.path.join(output_dir, f'data{num:04}.pickle')

    print(f"Writing data to: {output_file}")
    with open(output_file, 'wb') as f:
        pickle.dump(data, f)



def setup_mesh_for_viewer(mesh_file, sigma_t, g, albedo):

    mesh = TriangleMesh(mesh_file)

    mesh.mesh_positions = mesh.mesh.vertices.astype(np.float32).T
    max_pos = np.max(mesh.mesh_positions, 1)
    min_pos = np.min(mesh.mesh_positions, 1)
    diag = max_pos - min_pos
    max_pos += 0.1 * diag
    min_pos -= 0.1 * diag

    scene = create_mts_scene_from_mesh(mesh_file)
    constraint_kd_tree, sampled_p, sampled_n = vae.datapipeline.build_constraint_kdtree(mesh_file, sigma_t, g, albedo)
    sampled_n = VectorCloud(sampled_p, sampled_n, 0.5)
    sampled_p = PointCloud(sampled_p)

    return mesh, min_pos, max_pos, scene, constraint_kd_tree, sampled_p, sampled_n


def tangent_components_to_world(tx, ty, n):
    """Converts a tangent space directon tx, ty to world coordinates using the normal n (specific to the use in viewer)"""
    local_vector = np.array([tx, ty, np.maximum(1e-2, 1.0 - np.sqrt(tx ** 2 + ty ** 2))])
    local_vector = local_vector / np.sqrt(np.sum(local_vector ** 2))
    t1, t2 = onb_duff(n)
    return local_vector[0] * t1 + local_vector[1] * t2 - n * local_vector[2]


def get_coeff_trianglemesh(in_pos, in_dir, normal, poly_coeffs, poly_order, scale_factor,
                           min_pos, max_pos, res, prediction_space):
    """Converts poly coefficients to a triangle mesh"""
    extent = [min_pos[0], max_pos[0], min_pos[1], max_pos[1], min_pos[2], max_pos[2]]
    extent = (np.array(extent) * scale_factor).tolist()
    f_taylor = vae.utils.polynomial_to_voxel_grid_new(np.atleast_2d(in_pos) * scale_factor,
                                                  np.atleast_2d(-in_dir), normal, poly_coeffs, poly_order,
                                                  prediction_space, extent=extent, res=res)
    verts, faces, _, _ = measure.marching_cubes_lewiner(f_taylor, 0)
    verts = (verts / res) * (max_pos - min_pos) + min_pos
    return TriangleMesh(None, verts, faces)


def intersect_mesh(p, camera, mesh):
    """Point on mesh picking routine"""
    if not mesh:
        return None, None
    d = get_view_ray_dir(p, camera)
    intersector = mesh.mesh.ray
    its_loc, _, hit_triangles = intersector.intersects_location(
        camera.pos[np.newaxis, :], d[np.newaxis, :])
    face_normal = None
    if len(its_loc) > 0:
        dist_to_cam = np.sum((its_loc - camera.pos) ** 2, axis=1)
        min_dist_point = np.argmin(dist_to_cam)
        hit_triangle = hit_triangles[min_dist_point]
        vertices = mesh.mesh.faces[hit_triangle, :]
        its_loc = np.array(its_loc[min_dist_point, :], dtype=np.float32)
        barycentric = trimesh.triangles.points_to_barycentric(
            mesh.mesh.vertices[vertices][np.newaxis, :, :], np.atleast_2d(its_loc))
        vertex_normals = mesh.mesh.vertex_normals[vertices]
        face_normal = utils.math.normalize(np.sum(barycentric.T * vertex_normals, 0))
    else:
        its_loc = None
        face_normal = None
    return its_loc, face_normal


class AsynchronousTask:

    def __init__(self, nTasks, task_method, listener):
        self.nTasks = nTasks
        self.stopped = Value('i', 0)
        self.task_method = task_method
        self.listener = listener
        # thread = threading.Thread(target=self.run, args=())
        self.scene = listener.scene

        self.thread = Process(target=self.run, args=())
        self.thread.daemon = True
        self.thread.start()

    def run(self):
        for i in range(self.nTasks):
            if self.stopped.value:
                print('interupted')
                return
            self.listener.event_queue.put(self.task_method(i, self.nTasks))
