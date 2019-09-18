#!/usr/bin/env python3
import argparse
import glob
import os
import pickle
import sys
import time
from enum import Enum
import copy
from multiprocessing import Queue

import numpy as np
import skimage
import skimage.io
import tensorflow as tf
from matplotlib import cm
import tqdm

import mitsuba
import mitsuba.render
import nanogui
import utils.math
import vae.config
import vae.config_abs
import vae.config_angular
import vae.config_scatter
import vae.datapipeline
import vae.utils
from mitsuba.core import *
from nanogui import (Button, ComboBox,
                     GroupLayout, ImageView, Label,
                     PopupButton, Widget, Window, entypo, glfw)
from utils.experiments import load_config
from utils.gui import (FilteredListPanel, FilteredPopupListPanel,
                       LabeledSlider, add_checkbox)
from vae.global_config import (DATADIR3D, FIT_REGULARIZATION, OUTPUT3D,
                               RESOURCEDIR, SCENEDIR3D)

import vae.model
from vae.model import generate_new_samples, sample_outgoing_directions
from vae.predictors import (AbsorptionPredictor, AngularScatterPredictor,
                            ScatterPredictor)
from viewer.datasources import PointCloud, VectorCloud
from viewer.utils import *
import viewer.utils
from viewer.viewer import GLTexture, ViewerApp
from utils.printing import printr, printg
import utils.mtswrapper


class Mode(Enum):
    REF = 0
    PREDICTION = 1
    RECONSTRUCTION = 2
    POLYREF = 3
    POLYTRAIN = 4


class Scatter3DViewer(ViewerApp):
    """Viewer to visualize a given fixed voxelgrid"""

    def set_mesh(self, mesh_file):
        self.mesh_file = mesh_file
        self.mesh, self.min_pos, self.max_pos, self.scene, self.constraint_kd_tree, self.sampled_p, self.sampled_n = setup_mesh_for_viewer(
            mesh_file, self.sigma_t, self.g, self.albedo)
        self.shape = self.scene.getShapes()[0]
        self.picked_point = PointCloud(np.zeros((1, 3)))
        self.picked_point2 = PointCloud(np.zeros((1, 3)))

        self.training_points = None
        self.its_loc = None
        self.its_loc2 = None
        self.face_normal = None

        self.computed_poly = False

    def extract_mesh_polys(self):
        all_coeffs = []
        t0 = time.time()
        feat_name = self.absorption_config.shape_features_name
        for i in tqdm.tqdm(range(self.mesh.mesh_positions.shape[1])):
            pos = self.mesh.mesh_positions[:, i].ravel()
            normal = self.mesh.mesh_normal[:, i].ravel()

            cfg = copy.deepcopy(self.scatter_config)
            cfg.polynomial_space = 'LS'

            features = vae.model.extract_shape_features(cfg, pos, self.scene, self.constraint_kd_tree, -normal,
                                                        self.g, self.sigma_t, self.albedo, True, self.mesh,
                                                        normal, False, self.kdtree_threshold, self.fit_regularization, self.use_hard_surface_constraint)

            poly_coeffs = features['coeffs']
            coeffs_to_show = np.copy(poly_coeffs)

            thickness = None
            nor_hist = None
            if not self.visualize_gt_absorption:

                absorption = vae.model.estimate_absorption(self.session, pos, -normal, normal,
                                                           self.absorption_config, self.absorption_pred, self.feature_statistics, self.albedo,
                                                           self.sigma_t, self.g, self.eta, features)

                t0 = time.time()
                ref_pos_mts = vae.utils.mts_p(pos)
                ref_dir_mts = vae.utils.mts_v(-normal)
                kernel_eps = vae.utils.kernel_epsilon(self.g, self.sigma_t, self.albedo)
                scale_fac = float(vae.utils.get_poly_scale_factor(kernel_eps))
                thickness = mitsuba.render.Volpath3D.samplePolyThickness([float(c) for c in features['coeffs'].ravel()], ref_pos_mts, ref_dir_mts, False,
                                                                         scale_fac, 8,
                                                                         ref_pos_mts, ref_dir_mts, 123, 0.0, 3.0 * float(np.sqrt(kernel_eps)))
                thickness = np.array([[thickness]])

                if 'nor_constraints' in features:
                    nors = vae.utils.mts_to_np(features['nor_constraints'])
                    cos_theta = np.sum(normal * nors, axis=1)
                    nor_hist, _ = np.histogram(cos_theta, 4, [-1.01, 1.01])
                    nor_hist = nor_hist[None, :]
            else:
                medium = vae.utils.medium_params_list([mitsuba.core.Spectrum(self.albedo)],
                                                      [mitsuba.core.Spectrum(self.sigma_t)], [self.g], [self.eta])

                ref_pos_mts = vae.utils.mts_p(pos)
                ref_dir_mts = vae.utils.mts_v(-normal)
                in_dir_mts = vae.utils.mts_v(normal / np.sqrt(np.sum(normal ** 2)))
                kernel_eps = vae.utils.kernel_epsilon(self.g, self.sigma_t, self.albedo)
                t0 = time.time()

                coeffs, _, _ = mitsuba.render.Volpath3D.fitPolynomials(ref_pos_mts, in_dir_mts, vae.utils.mts_v(normal),
                                                                       vae.utils.kernel_epsilon(
                    self.g, self.sigma_t, self.albedo),
                    self.fit_regularization, self.scatter_config.poly_order(), 'gaussian',
                    self.kdtree_threshold, self.constraint_kd_tree, False, self.sigma_t, self.use_hard_surface_constraint)

                coeffs = np.array(coeffs)
                # coeffs[0] = 0.0  # Polygon should really be zero on the current position
                seed = int(np.random.randint(10000))
                scale_fac = float(vae.utils.get_poly_scale_factor(kernel_eps))
                tmp_result = mitsuba.render.Volpath3D.samplePolyFixedStart([float(c) for c in coeffs], ref_pos_mts, ref_dir_mts,
                                                                           False, scale_fac, medium, 128, 128, 256,
                                                                           ref_pos_mts, ref_dir_mts, self.ignore_zero_scatter,
                                                                           self.disable_rr, seed)
                absorption = np.mean(tmp_result['absorptionProb'])
                absorption = np.array([[absorption]])

                t0 = time.time()
                thickness = mitsuba.render.Volpath3D.samplePolyThickness([float(c) for c in coeffs], ref_pos_mts, ref_dir_mts, False,
                                                                         scale_fac, 8,
                                                                         ref_pos_mts, ref_dir_mts, seed, 0.0, 3.0 * float(kernel_eps))
                thickness = np.array([[thickness]])

            feat_to_show = [1.0 - absorption]
            if False:  # For now, do not visualize any of these extra features
                if nor_hist is not None:
                    feat_to_show.append(nor_hist)
                if thickness is not None:
                    feat_to_show.append(thickness)
            feat_to_show.append(coeffs_to_show)
            all_coeffs.append(np.concatenate(feat_to_show, 1))

        print(f'Took {time.time() - t0} s')
        self.mesh_polys = np.concatenate(all_coeffs, 0)
        print(f"self.mesh_polys.shape: {self.mesh_polys.shape}")

    def set_dataset(self, dataset):
        self.dataset = vae.datapipeline.get_config_from_metadata(dataset, OUTPUT3D)
        self.dataset = self.dataset(SCENEDIR3D, RESOURCEDIR, os.path.join(DATADIR3D, dataset))
        # Get mesh generator and update mesh list
        self.mesh_files, _ = self.dataset.mesh_generator.get_meshes(False)
        self.set_mesh(self.mesh_files[0])
        # get the medium param generator and set sliders accordingly
        med_gen = self.dataset.medium_param_generator
        med = med_gen.get_media(1)
        self.albedo_slider.set_value(med[0].albedo[0])
        self.sigma_t_slider.set_value(med[0].sigmaT[0])
        self.g_slider.set_value(med[0].g)
        self.eta_slider.set_value(med[0].ior)
        self.performLayout()

    def __init__(self, args):
        super(Scatter3DViewer, self).__init__()

        parser = argparse.ArgumentParser(description='''SSS Viewer''')
        parser.add_argument('--mesh', default=None)
        parser.add_argument('--net', default=None)
        parser.add_argument('--absnet', default=None)
        parser.add_argument('-n', default=100000, type=int)
        args = parser.parse_args(args)

        self.pos_constraints_pc = None

        self.absorption_config, self.scatter_config, self.angular_scatter_config = None, None, None
        self.viewer_output_dir = os.path.join(OUTPUT3D, 'viewer')
        os.makedirs(self.viewer_output_dir, exist_ok=True)

        self.sigma_t, self.albedo, self.g, self.eta = 1.0, 0.75, 0.0, 1.0

        self.datasets = os.listdir(DATADIR3D)
        self.dataset = vae.datapipeline.get_config_from_metadata(self.datasets[0], OUTPUT3D)
        self.dataset = self.dataset(SCENEDIR3D, RESOURCEDIR, os.path.join(DATADIR3D, self.datasets[0]))

        self.mesh_files, _ = self.dataset.mesh_generator.get_meshes(False)

        self.viewer_data, self.training_points, self.its_loc = None, None, None
        self.inDirection, self.inDirectionViz, self.face_normal = None, None, None

        self.mesh_idx = 0

        if args.mesh and os.path.isfile(args.mesh):
            self.set_mesh(args.mesh)
        else:
            self.set_mesh(self.mesh_files[self.mesh_idx])
        self.extra_point = None
        self.extra_dir = None
        self.rec_mesh = None
        self.main_window = Window(self, "Scatter3DViewer")
        self.main_window.setPosition((15, 15))
        self.main_window.setLayout(GroupLayout())

        vscroll = nanogui.VScrollPanel(self.main_window)
        vscroll.setFixedHeight(900)
        vscroll.setFixedWidth(300)
        tools = Widget(vscroll)
        tools.setLayout(GroupLayout())
        self.tangent_x, self.tangent_y = 0, 0
        self.mode = Mode.REF

        self.recording_samples = False
        self.stored_samples = []

        def update_mesh_idx():
            self.set_mesh(self.mesh_files[self.mesh_idx])
            return True

        def update_medium():
            its_loc, its_loc2 = self.its_loc, self.its_loc2
            face_normal = self.face_normal
            self.set_mesh(self.mesh_file)  # reset mesh to regen constraint kdtree
            self.its_loc, self.its_loc2 = its_loc, its_loc2
            self.face_normal = face_normal
            self.update_displayed_scattering()
            return True

        Button(tools, "Open mesh").setCallback(lambda: self.set_mesh(nanogui.file_dialog([("obj", "3D Object")], True)))
        self.albedo_slider = LabeledSlider(self, tools, 'albedo', 0.05, 1, float,
                                           update_medium, slider_width=160,
                                           warp_fun=vae.utils.albedo_to_effective_albedo, inv_warp_fun=vae.utils.effective_albedo_to_albedo)
        self.sigma_t_slider = LabeledSlider(self, tools, 'sigma_t', 0.01, 1200, float,
                                            update_medium, True, slider_width=160)
        self.g_slider = LabeledSlider(self, tools, 'g', -0.999, 0.999, float,
                                      update_medium, slider_width=160)
        self.eta_slider = LabeledSlider(self, tools, 'eta', 1.0, 3.5, float,
                                        update_medium, slider_width=160)
        self.tangent_x_slider = LabeledSlider(self, tools, 'tangent_x', -1, 1, float,
                                              self.update_displayed_scattering, slider_width=160)
        self.tangent_y_slider = LabeledSlider(self, tools, 'tangent_y', -1, 1, float,
                                              self.update_displayed_scattering, slider_width=160)

        def cb_compute_poly():
            self.computed_poly = False
            self.update_displayed_scattering()
        self.kdtree_threshold = 0.0
        self.kdtree_threshold_slider = LabeledSlider(self, tools, 'kdtree_threshold', 0.0, 1.0, float,
                                                     cb_compute_poly, slider_width=160)
        self.fit_regularization = FIT_REGULARIZATION
        self.fit_regularization_slider = LabeledSlider(self, tools, 'fit_regularization', 0.0, 100.0, float,
                                                       cb_compute_poly, slider_width=160)
        self.mesh_idx_slider = LabeledSlider(self, tools, 'mesh_idx', 0, 100, int, update_mesh_idx, slider_width=160)

        self.n_scatter_samples = 4096
        LabeledSlider(self, tools, 'n_scatter_samples', 32, 2 ** 19, int, self.update_displayed_scattering, slider_width=160,
                      warp_fun=np.log2, inv_warp_fun=lambda x: 2 ** x)

        add_checkbox(self, tools, 'project_samples', True, label='Project Points')
        self.show_rec_mesh_checkbox = add_checkbox(self, tools, 'show_rec_mesh', False, label='Show Reconstructed Mesh')
        add_checkbox(self, tools, 'show_histograms', False, label='Show Histograms')
        add_checkbox(self, tools, 'show_poly_coeffs', False, label='Show Poly Coefficients')
        self.poly_coeff_to_show = 0
        self.poly_coeff_show_slider = LabeledSlider(self, tools, 'poly_coeff_to_show', 0, 20, int, slider_width=160)

        add_checkbox(self, tools, 'show_fit_kernel', False, label='Show Fit Kernel')
        add_checkbox(self, tools, 'use_legacy_epsilon', False,
                     self.update_displayed_scattering, label='Use Legacy Fit Epsilon')
        add_checkbox(self, tools, 'use_hard_surface_constraint', True,
                     self.update_displayed_scattering, label='Use hard surface constraint')
        add_checkbox(self, tools, 'use_svd', False, self.update_displayed_scattering, label='Use SVD')
        add_checkbox(self, tools, 'use_mesh_histogram', False, label='Mesh Histograms')
        add_checkbox(self, tools, 'occlusion_culling', False, label='Cull Points')

        popupBtn = PopupButton(tools, "Extra Settings", entypo.ICON_EXPORT)
        popup = popupBtn.popup()
        popup.setLayout(GroupLayout())
        add_checkbox(self, popup, 'print_camera_position', False, cb_compute_poly, label='Print camera position to CMD')
        add_checkbox(self, popup, 'visualize_gt_absorption', False,
                     cb_compute_poly, label='Visualize Groundtruth Absorption')
        add_checkbox(self, popup, 'show_no_gd_samples', False,
                     self.update_displayed_scattering, label='show_no_gd_samples')
        add_checkbox(self, popup, 'predict_absorption', False,
                     self.update_displayed_scattering, label='Run the absorption prediction')
        add_checkbox(self, popup, 'show_projection_debug_info', False, label='Show Projection Debug Info')
        add_checkbox(self, popup, 'show_off_surface_error', False,
                     self.update_displayed_scattering, label='Show Off-Surface Error')
        add_checkbox(self, popup, 'autoload_dataset', args.mesh is None,
                     label='Automatically load the dataset associated with a model')
        add_checkbox(self, popup, 'disable_shape_features', False,
                     self.update_displayed_scattering, label='Disable Shape Features')
        add_checkbox(self, popup, 'rotate_poly', True, self.update_displayed_scattering, label='Poly Rotation')

        add_checkbox(self, popup, 'disable_rr', False,
                     self.update_displayed_scattering, label='Disable Russian Roulette')
        add_checkbox(self, popup, 'ignore_zero_scatter', True, self.update_displayed_scattering)
        add_checkbox(self, popup, 'show_sampled_points', False, label='Show Sampled Surface Points')
        add_checkbox(self, popup, 'show_used_constraints', False, label='Show Used Constraint Points')
        add_checkbox(self, popup, 'show_scatter_points', True, label='Show Sampled Outgoing Positions')
        add_checkbox(self, popup, 'show_nbounces_histogram', False,
                     lambda: self.nbounces_window.setVisible(self.show_nbounces_histogram),
                     label='Show N-Bounces Histogram')
        add_checkbox(self, popup, 'importance_sample_train_data', False, label='Importance Sampling Training Points')

        self.setBackground(nanogui.Color(0.2, 0.2, 0.2, 1.0))

        def cb():
            if self.white_background:
                self.setBackground(nanogui.Color(1.0, 1.0, 1.0, 1.0))
            else:
                self.setBackground(nanogui.Color(0.2, 0.2, 0.2, 1.0))
        add_checkbox(self, popup, 'white_background', False, cb, label='White Background')
        add_checkbox(self, popup, 'show_grid', True, cb, label='Show Grid')

        popupBtn = PopupButton(tools, "Angular Scattering", entypo.ICON_EXPORT)
        popup = popupBtn.popup()
        popup.setLayout(GroupLayout())

        def cb():
            self.angular_histogram_window.setVisible(self.show_angular_scattering)

        add_checkbox(self, popup, 'sample_outdirs', False, cb, label='Sample outgoing directions')
        add_checkbox(self, popup, 'show_angular_scattering', False, cb, label='Show Angular Histogram')
        add_checkbox(self, popup, 'visualize_histogram_radius', False, None, label='Show Histogram Radius')
        add_checkbox(self, popup, 'show_outgoing_dir', False, label='Show Outgoing Directions')

        self.angular_histogram_parametrization = AngularParametrization.CONCENTRIC

        def cb(value):
            a = AngularParametrization
            self.angular_histogram_parametrization = [a.CONCENTRIC,
                                                      a.PROJECTION, a.POLAR, a.WORLDSPACE, a.CONCENTRIC_RESCALED][value]
            self.update_angular_histogram()

        Label(popup, 'Parametrization')
        parametrization_combobox = ComboBox(popup)
        parametrization_combobox.setItems(['Concentric', 'Projection', 'Polar', 'World Space', 'Concentric Rescaled'])
        parametrization_combobox.setCallback(cb)

        self.angular_histogram_radius = 0.5
        LabeledSlider(self, popup, 'angular_histogram_radius', 0, 5, float,
                      self.update_angular_histogram, slider_width=120)

        self.angular_min_bounces = 0
        self.angular_max_bounces = -1
        LabeledSlider(self, popup, 'angular_min_bounces', 0, 10, int,
                      self.update_angular_histogram, slider_width=120)
        LabeledSlider(self, popup, 'angular_max_bounces', -1, 10, int,
                      self.update_angular_histogram, slider_width=120)

        Button(popup, "Save Histogram").setCallback(
            lambda: self.save_angular_histogram(nanogui.file_dialog([("png", "Image")], True)))

        def cb(value):
            self.mode = [Mode.REF, Mode.PREDICTION, Mode.RECONSTRUCTION,
                         Mode.POLYREF, Mode.POLYTRAIN][value]
            self.update_displayed_scattering()

        Label(tools, 'Mode')
        self.mode_combobox = ComboBox(tools)
        self.mode_combobox.setItems(['Reference', 'Prediction', 'Reconstruction',
                                     'Poly Reference', 'Poly Train'])
        self.mode_combobox.setCallback(cb)

        Label(tools, 'Data Set')
        self.data_set_combobox = ComboBox(tools)
        self.data_set_combobox.setItems(self.datasets)

        def cb_dataset(value):
            self.set_dataset(self.datasets[value])

        self.data_set_combobox.setCallback(cb_dataset)

        self.networks = sorted(glob.glob(os.path.join(OUTPUT3D, 'models', '*')))

        if args.n < len(self.networks):
            self.networks = self.networks[-args.n:]

        self.absorption_networks = sorted(glob.glob(os.path.join(OUTPUT3D, 'models_abs', '*')))
        self.angular_networks = sorted(glob.glob(os.path.join(OUTPUT3D, 'models_angular', '*')))
        self.networks_short = [os.path.split(n)[-1] for n in self.networks]
        self.absorption_networks_short = [os.path.split(n)[-1] for n in self.absorption_networks]
        self.angular_networks_short = [os.path.split(n)[-1] for n in self.angular_networks]
        self.session = None

        self.scatter_net = self.networks[-1]
        self.absorption_net = self.absorption_networks[-1]
        self.angular_scatter_net = self.angular_networks[-1]

        if args.net:
            self.scatter_net = os.path.join(OUTPUT3D, 'models', args.net)
        if args.absnet:
            self.absorption_net = os.path.join(OUTPUT3D, 'models_abs', args.absnet)
            self.predict_absorption = True

        self.load_networks()

        def cb(value):
            self.scatter_net = self.networks[value]
            self.load_networks()
            self.update_displayed_scattering()
        Label(tools, 'Scatter Network')
        popoutList = FilteredPopupListPanel(tools, self.networks_short, self)
        popoutList.setCallback(cb)
        popoutList.setSelectedIndex(len(self.networks_short) - 1)

        def cb(value):
            self.absorption_net = self.absorption_networks[value]
            self.load_networks()
            self.update_displayed_scattering()
        Label(tools, 'Absorption Network')
        popoutList = FilteredPopupListPanel(tools, self.absorption_networks_short, self)
        popoutList.setCallback(cb)
        popoutList.setSelectedIndex(len(self.absorption_networks_short) - 1)

        def cb(value):
            self.angular_scatter_net = self.angular_networks[value]
            self.load_networks()
            self.update_displayed_scattering()
        Label(tools, 'Angular Scatter Network')
        popoutList = FilteredPopupListPanel(tools, self.angular_networks_short, self)
        popoutList.setCallback(cb)
        popoutList.setSelectedIndex(len(self.angular_networks_short) - 1)

        self.nbounces_window = Window(self, "Bounces")
        self.nbounces_window.setPosition((350, 15))
        self.nbounces_window.setLayout(GroupLayout())
        Label(self.nbounces_window, "Histogram of number of bounces", "sans-bold")
        self.n_bounces_graph = nanogui.Graph(self.nbounces_window, "")
        self.n_bounces_graph.setWidth(250)
        self.n_bounces_graph.setFixedHeight(100)
        self.nbounces_window.setVisible(self.show_nbounces_histogram)

        self.angular_histogram_window = Window(self, "Histogram of outgoing directions")
        self.angular_histogram_window.setPosition((600, 15))
        self.angular_histogram_window.setLayout(GroupLayout())
        self.angular_histogram_window.setVisible(self.show_angular_scattering)

        img_data = np.zeros((256, 256, 3))
        self.img = GLTexture(img_data)
        self.angular_img_view = ImageView(self.angular_histogram_window, self.img.id)
        self.angular_img_view.setGridThreshold(3)

        self.performLayout()
        self.thread_count = 0
        self.t = None
        self.sampling_task = None

    def exitEvent(self):
        if self.sampling_task:
            if self.sampling_task.thread.is_alive:
                self.sampling_task.thread.terminate()
                self.sampling_task.thread.join()
        super().exitEvent()

    def load_networks(self):
        self.session = None
        tf.reset_default_graph()
        vae.model.recreate_shared_functions()

        loaded_scatter_config = load_config(self.scatter_net)

        sub_configs = loaded_scatter_config['args']['config'].split('/')
        separate_abs_model, separate_angular_model = True, True
        scatter_config_name = sub_configs[0]
        if len(sub_configs) > 1:
            abs_config_name = sub_configs[1]
            separate_abs_model = False
            self.predict_absorption = True
            if len(sub_configs) > 2:
                angular_config_name = sub_configs[2]
                separate_angular_model = False

        if separate_abs_model:
            loaded_abs_config = load_config(self.absorption_net)
            abs_config_name = loaded_abs_config['args']['config']

        if separate_angular_model:
            loaded_angular_config = load_config(self.angular_scatter_net)
            angular_config_name = loaded_angular_config['args']['config']

        self.ph_manager = vae.predictors.PlaceholderManager(dim=3)

        self.scatter_config = vae.config.get_config(vae.config_scatter, scatter_config_name)
        # Load the dataset used for training this model
        self.scatter_config.dataset = loaded_scatter_config['config0']['dataset']
        self.scatter_config.dim = 3
        self.scatter_pred = ScatterPredictor(self.ph_manager, self.scatter_config, None)
        if self.autoload_dataset:
            self.set_dataset(self.scatter_config.dataset)

        self.absorption_config = vae.config.get_config(vae.config_abs, abs_config_name)
        self.absorption_config.dim = 3
        if self.predict_absorption:
            self.absorption_pred = AbsorptionPredictor(self.ph_manager, self.absorption_config, None)

        self.angular_scatter_config = vae.config.get_config(vae.config_angular, angular_config_name)
        self.angular_scatter_config.dim = 3
        if self.sample_outdirs:
            self.angular_scatter_pred = AngularScatterPredictor(self.ph_manager, self.angular_scatter_config, None)

        if self.session is not None:
            self.session.close()
        # self.session = tf.Session(config=tf.ConfigProto(
        #     gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.333)))

        self.session = tf.Session(config=tf.ConfigProto(
            gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.333)))
        # Restore all three networks into the same session

        if not separate_abs_model and not (separate_angular_model and self.sample_outdirs):
            vae.tf_utils.restore_model(self.session, self.scatter_net, None)
        else:
            vae.tf_utils.restore_model(self.session, self.scatter_net, self.scatter_pred.prefix)
        if self.predict_absorption and separate_abs_model:
            vae.tf_utils.restore_model(self.session, self.absorption_net, self.absorption_pred.prefix)
        if self.sample_outdirs and separate_angular_model:
            vae.tf_utils.restore_model(self.session, self.angular_scatter_net, self.angular_scatter_pred.prefix)

        # Assumes that all 3 models are trained on the same data with same feature statistics
        dataset = vae.datapipeline.get_config_from_metadata(self.scatter_config.dataset, OUTPUT3D)(
            SCENEDIR3D, RESOURCEDIR, os.path.join(DATADIR3D, self.scatter_config.dataset))
        self.feature_statistics = dataset.get_feature_statistics()

    def save_angular_histogram(self, path):
        print(f"Saving angular histogram to {path}")
        skimage.io.imsave(path, self.raw_angular_hist_data / np.max(self.raw_angular_hist_data))

    def update_angular_histogram(self):
        if self.its_loc2 is None:
            return
        hist_data = self.viewer_data.get_angular_histogram(
            self.its_loc2, self.angular_histogram_radius, self.angular_histogram_parametrization, self.eta,
            self.angular_min_bounces, self.angular_max_bounces)

        hist_data = hist_data / np.max(hist_data)
        self.raw_angular_hist_data = np.copy(hist_data)
        hist_data = skimage.transform.resize(hist_data, [256, 256], order=0, mode='constant')
        hist_data = cm.jet(hist_data)

        # Draw circle marking the critical angle on top
        theta_crit = np.arcsin(1.0 / self.eta)
        r = 1 / self.eta

        xx, yy = 2.0 * np.mgrid[:256, :256] / 256 - 1.0

        circle = xx ** 2 + yy ** 2
        thickness = 0.01
        circle = np.logical_and(circle < (r ** 2 + thickness), circle > (r ** 2 - thickness)).astype(np.float32)
        if self.angular_histogram_parametrization != AngularParametrization.CONCENTRIC_RESCALED:
            hist_data[:, :, 0] = np.maximum(hist_data[:, :, 0], circle)

        self.img = GLTexture(hist_data)
        self.angular_img_view.bindImage(self.img.id)

    def handleEvent(self, event):

        if event['mode'] == Mode.PREDICTION:
            self.viewer_data.append(event['outPos'],
                                    points_unproj=event['outPosUnproj'],
                                    normals=event['outNormal'],
                                    out_dirs=event['outDir'])
        else:
            self.viewer_data.append(event['outPos'],
                                    normals=event['outNormal'],
                                    out_dirs=event['outDir'],
                                    n_bounces=event['nBounces'])
        self.update_angular_histogram()

    def get_poly_fit_options(self):
        return {'regularization': self.fit_regularization, 'useSvd': self.use_svd, 'globalConstraintWeight': 0.01,
                'order': self.scatter_config.poly_order(), 'kdTreeThreshold': self.kdtree_threshold,
                'useSimilarityKernel': not self.use_legacy_epsilon, 'useLightspace': False}

    def update_displayed_scattering(self):
        if self.its_loc is None or self.face_normal is None:
            return
        print(f"self.its_loc: {self.its_loc}")
        print(f"self.face_normal: {self.face_normal}")
        if self.print_camera_position:
            print(f"self.camera.pos: {self.camera.pos}")

        sigma_tp = vae.utils.get_sigmatp(self.albedo, self.g, self.sigma_t)
        printg(f"sigma_tp: {sigma_tp}")

        scale_factor = vae.utils.get_poly_scale_factor(vae.utils.kernel_epsilon(self.g, self.sigma_t, self.albedo))

        self.inDirection = tangent_components_to_world(self.tangent_x, self.tangent_y, self.face_normal)
        self.inDirectionViz = VectorCloud(self.its_loc, -self.inDirection)
        self.training_points = None
        chunk_size = 4096
        n_chunks = self.n_scatter_samples // chunk_size
        sample_counts = [chunk_size] * n_chunks
        if self.n_scatter_samples % chunk_size:
            sample_counts.append(self.n_scatter_samples % chunk_size)
        medium = vae.utils.medium_params_list([mitsuba.core.Spectrum(self.albedo)], [
            mitsuba.core.Spectrum(self.sigma_t)], [self.g], [self.eta])
        if self.mode == Mode.REF:
            t0 = time.time()
            self.viewer_data = ViewerState(self)
            seed = int(np.random.randint(10000))

            def f(k, n):
                mts_result = mitsuba.render.Volpath3D.sampleFixedStart(self.scene, self.shape, medium, sample_counts[k], sample_counts[k],
                                                                       sample_counts[k], vae.utils.mts_p(self.its_loc),
                                                                       vae.utils.mts_v(
                                                                           self.inDirection), self.ignore_zero_scatter,
                                                                       self.disable_rr, seed + k)
                print(f"mts_result['absorptionProb'][0]: {mts_result['absorptionProb'][0]}")
                result = {'mode': Mode.REF,
                          'outPos': vae.utils.mts_to_np(mts_result['outPos']),
                          'outNormal': vae.utils.mts_to_np(mts_result['outNormal']),
                          'outDir': vae.utils.mts_to_np(mts_result['outDir']),
                          'nBounces': np.array(mts_result['bounces'])}
                return result

            if self.sampling_task:
                self.sampling_task.stopped.value = True
                self.sampling_task.thread.terminate()
                self.sampling_task.thread.join(0.001)
                self.event_queue = Queue()
            self.sampling_task = AsynchronousTask(len(sample_counts), f, self)

        elif self.mode == Mode.PREDICTION:
            self.viewer_data = ViewerState(self, need_projection=True,
                                           poly_order=self.scatter_config.poly_order(),
                                           prediction_space=self.scatter_config.prediction_space)

            def f(k, n):
                scale_factor = vae.utils.get_poly_scale_factor(vae.utils.kernel_epsilon(
                    self.g, self.sigma_t, self.albedo, self.use_legacy_epsilon))

                # coeffs = np.zeros((1, 20) ,dtype=np.float32)
                # coeffs[:, 3] = 1.0 / scale_factor
                # coeffs[:, 9] = 1.0 / scale_factor**2
                # features = {'features': 'poly', 'coeffs': coeffs}
                features = None

                print(f"scale_factor: {scale_factor}")
                generated_samples, generated_samples2, extra_info = generate_new_samples(self.session, self.its_loc, self.inDirection,
                                                                                         self.face_normal, self.mesh_files[self.mesh_idx],
                                                                                         self.scatter_config, self.scatter_pred, self.feature_statistics,
                                                                                         self.n_scatter_samples, self.albedo,
                                                                                         self.sigma_t, self.g, self.eta, self.constraint_kd_tree,
                                                                                         self.disable_shape_features, self.rotate_poly,
                                                                                         self.use_legacy_epsilon, self.kdtree_threshold, self.fit_regularization,
                                                                                         self.scene, self.use_hard_surface_constraint, features=features)
                coeffs = None
                if self.scatter_config.use_sh_coeffs:
                    outPos = np.copy(generated_samples)
                    outNormal = outPos
                    valid_pos = np.ones(outPos.shape[0], dtype=np.bool)
                    if 'coeffs' in extra_info and extra_info['coeffs'] is not None:
                        shCoeffs = extra_info['coeffs']

                if 'coeffs' in extra_info and not self.scatter_config.use_sh_coeffs and extra_info['coeffs'] is not None:
                    outPos, outNormal, valid_pos = utils.mtswrapper.project_points_on_mesh(self.scene, generated_samples,
                                                                                           self.its_loc, -self.inDirection, extra_info['coeffs_ws'],
                                                                                           self.scatter_config.poly_order(),
                                                                                           False, scale_factor, False)
                if self.predict_absorption:
                    absorption = vae.model.estimate_absorption(self.session, self.its_loc, self.inDirection, self.face_normal,
                                                               self.absorption_config, self.absorption_pred, self.feature_statistics, self.albedo,
                                                               self.sigma_t, self.g, self.eta, extra_info)
                    print(f"Absorption self.its_loc: {self.its_loc}")
                    print(f"absorption: {absorption.ravel()[0]}")

                # Evaluate polynomial gradient

                if self.show_no_gd_samples:
                    generated_samples = generated_samples2

                if coeffs is not None:
                    poly_value = vae.utils.eval_poly(generated_samples, self.its_loc, -
                                                     self.inDirection, coeffs, True, scale=scale_factor)
                    poly_grad = vae.utils.eval_poly_gradient(
                        generated_samples, self.its_loc, -self.inDirection, coeffs, True, scale=scale_factor)
                    poly_normal = utils.math.normalize(poly_grad)[valid_pos]
                else:
                    poly_value = None
                use_geom_normals = False
                if self.sample_outdirs:
                    points_unscaled = self.sigma_t * (outPos[valid_pos] - self.its_loc) + self.its_loc
                    out_dirs = sample_outgoing_directions(self.session, self.its_loc, self.inDirection, self.face_normal,
                                                          points_unscaled, outNormal[valid_pos] if use_geom_normals else poly_normal,
                                                          self.angular_scatter_config, self.angular_scatter_pred, self.feature_statistics,
                                                          points_unscaled.shape[0], self.dataset, self.albedo,
                                                          self.sigma_t, self.g, self.eta, self.disable_shape_features)
                else:
                    out_dirs = None

                if 'pos_constraints' in extra_info and extra_info['pos_constraints'] is not None:
                    extra_info['pos_constraints'] = vae.utils.mts_to_np(extra_info['pos_constraints'])

                result = {'mode': Mode.PREDICTION,
                          'outPos': outPos[valid_pos],
                          'outPosUnproj': generated_samples,
                          'outNormal': outNormal[valid_pos],
                          'polyValue': poly_value,
                          'outDir':  out_dirs,
                          'coeffs': coeffs,
                          'posConstraints': extra_info['pos_constraints'] if 'pos_constraints' in extra_info else None}
                return result

            if self.sampling_task:
                self.sampling_task.stopped.value = True
                self.sampling_task.thread.terminate()
                self.sampling_task.thread.join(0.001)
                self.event_queue = Queue()
            # self.sampling_task = AsynchronousTask(4, f, self)
            # High absorption 0.91
            #true_its_loc = self.its_loc
            #self.its_loc = np.array([-3.9024332, 3.6581001, 3.98363])
            #result = f(0, 0)

           # Low absorption
            #self.its_loc = np.array([-3.897561, 3.6556835, 3.98363])
            #result = f(0, 0)
            #self.its_loc = true_its_loc
            result = f(0, 0)
            self.viewer_data.coeffs = result['coeffs']

            if 'posConstraints' in result and result['posConstraints'] is not None:
                print(f"len(result['posConstraints']): {len(result['posConstraints'])}")
                self.pos_constraints_pc = PointCloud(result['posConstraints'])

            if self.show_off_surface_error:
                result['polyValue'] = np.abs(result['polyValue'])
                result['polyValue'] = result['polyValue'] / np.max(result['polyValue'])
                self.viewer_data.append(result['outPos'],
                                        unproj_points=result['outPosUnproj'],
                                        normals=result['outNormal'],
                                        out_dirs=result['outDir'],
                                        point_unproj_colors=np.ones_like(result['outPosUnproj']) * result['polyValue'])
            else:
                self.viewer_data.append(result['outPos'],
                                        unproj_points=result['outPosUnproj'],
                                        normals=result['outNormal'],
                                        out_dirs=result['outDir'])
        elif self.mode == Mode.RECONSTRUCTION:
            t0 = time.time()
            self.viewer_data = ViewerState(self)
            seed = int(np.random.randint(10000))

            # 1. Sample points using polynomial VPT
            ref_pos_mts = vae.utils.mts_p(self.its_loc)
            ref_dir_mts = vae.utils.mts_v(self.inDirection)
            in_dir_mts = vae.utils.mts_v(-self.inDirection / np.sqrt(np.sum(self.inDirection ** 2)))

            fit_opts = self.get_poly_fit_options()
            fit_opts['useLightspace'] = False
            coeffs, _, _ = utils.mtswrapper.fitPolynomial(self.constraint_kd_tree, self.its_loc, -self.inDirection, self.sigma_t,
                                                          self.g, self.albedo, fit_opts, normal=self.face_normal)
            fit_opts['useLightspace'] = True
            coeffs_ls, _, _ = utils.mtswrapper.fitPolynomial(self.constraint_kd_tree, self.its_loc, -self.inDirection, self.sigma_t,
                                                             self.g, self.albedo, fit_opts, normal=self.face_normal)
            t0 = time.time()
            tmp_result = mitsuba.render.Volpath3D.samplePolyFixedStart([float(c) for c in coeffs], ref_pos_mts, ref_dir_mts,
                                                                       False, scale_factor, medium, self.n_scatter_samples, 1, 1,
                                                                       ref_pos_mts, ref_dir_mts, self.ignore_zero_scatter,
                                                                       self.disable_rr, seed)
            gt_absorption = np.mean(tmp_result['absorptionProb'])
            outPos = vae.utils.mts_to_np(tmp_result['outPos'])
            print('Took {} s'.format(time.time() - t0))
            print('absorption: {}'.format(gt_absorption))

            # 2. Try to reconstruct them using the VAE: Are they far from the surface?
            reconstructed_samples = vae.model.vae_reconstruct_samples(self.session, outPos, coeffs_ls, self.its_loc, self.inDirection, self.face_normal,
                                                                      self.scatter_config, self.scatter_pred, self.feature_statistics,
                                                                      self.albedo, self.sigma_t, self.g, self.eta, self.disable_shape_features)

            self.viewer_data = ViewerState(self, need_projection=True, poly_order=self.scatter_config.poly_order(), prediction_space=self.scatter_config.prediction_space)
            self.viewer_data.coeffs = coeffs_ls
            self.viewer_data.append(reconstructed_samples,
                                    normals=vae.utils.mts_to_np(tmp_result['outNormal']),
                                    out_dirs=vae.utils.mts_to_np(tmp_result['outDir']))

        elif self.mode == Mode.POLYREF:

            print(f"self.get_poly_fit_options(): {self.get_poly_fit_options()}")

            coeffs, pos_constraints, _ = utils.mtswrapper.fitPolynomial(self.constraint_kd_tree,
                                                                        self.its_loc,
                                                                        -self.inDirection,
                                                                        self.sigma_t,
                                                                        self.g,
                                                                        self.albedo,
                                                                        self.get_poly_fit_options(),
                                                                        normal=self.face_normal)

            self.pos_constraints_pc = PointCloud(pos_constraints)
            self.viewer_data = ViewerState(self, need_projection=True, poly_order=self.scatter_config.poly_order(),
                                           prediction_space='WS')
            # coeffs[0] = 0.0  # Polygon should really be zero on the current position
            self.viewer_data.coeffs = coeffs
            seed = int(np.random.randint(10000))

            kernel_eps = vae.utils.kernel_epsilon(self.g, self.sigma_t, self.albedo)
            scale_fac = float(vae.utils.get_poly_scale_factor(kernel_eps))
            ref_pos_mts = vae.utils.mts_p(self.its_loc)
            ref_dir_mts = vae.utils.mts_v(self.inDirection)

            l = mitsuba.render.Volpath3D.adjustRayForPolynomialTracing(ref_pos_mts, ref_dir_mts, [float(c) for c in coeffs], ref_pos_mts, ref_dir_mts,
                                                                       False, scale_fac, vae.utils.mts_v(self.face_normal))

            success = l[0]
            if not success:
                print('Failed adjusting polynomial')
            trace_pos_mts = l[1]
            trace_dir_mts = l[2]

            trace_dir = vae.utils.mts_to_np(trace_dir_mts)
            trace_dir = self.inDirection

            poly_val = vae.utils.eval_poly(vae.utils.mts_to_np(trace_pos_mts),
                                           self.its_loc, self.its_loc, coeffs, False)
            poly_grad = utils.math.normalize(vae.utils.eval_poly_gradient(
                self.its_loc, self.its_loc, self.its_loc, coeffs, False))
            if not success:
                self.extra_point = PointCloud(self.its_loc, colors=np.array([[1.0, 1.0, 0.0]]))
                self.extra_dir = VectorCloud(self.its_loc, -5 * poly_grad)
            else:
                self.extra_point = PointCloud(vae.utils.mts_to_np(trace_pos_mts), colors=np.array([[1.0, 1.0, 0.0]]))
                self.extra_dir = VectorCloud(vae.utils.mts_to_np(trace_pos_mts), -vae.utils.mts_to_np(trace_dir_mts))

            def f(k, n):
                t0 = time.time()
                tmp_result = utils.mtswrapper.sample_scattering_poly2(
                    self.its_loc, trace_dir, sample_counts[k], 1, 1, medium, coeffs, seed + k, {})
                print('Took {} s'.format(time.time() - t0))
                gt_absorption = np.mean(tmp_result['absorptionProb'])
                print('absorption: {}'.format(gt_absorption))
                result = {'mode': Mode.POLYREF,
                          'outPos': vae.utils.mts_to_np(tmp_result['outPos']),
                          'outNormal': vae.utils.mts_to_np(tmp_result['outNormal']),
                          'outDir': vae.utils.mts_to_np(tmp_result['outDir']),
                          'nBounces': np.array(tmp_result['bounces']),
                          'coeffs': coeffs}
                return result

            if self.sampling_task:
                self.sampling_task.stopped.value = True
                self.sampling_task.thread.terminate()
                self.sampling_task.thread.join(0.001)
                self.event_queue = Queue()
            self.sampling_task = AsynchronousTask(len(sample_counts), f, self)

        elif self.mode == Mode.POLYTRAIN:
            t0 = time.time()
            batch_size = 64
            n_abs_samples = 64
            seed = int(np.random.randint(10000))

            opts = self.get_poly_fit_options()
            opts['disable_rr'] = self.disable_rr
            opts['importance_sample_polys'] = self.importance_sample_train_data
            tmp_result = utils.mtswrapper.sample_scattering_mesh(self.mesh_file, self.n_scatter_samples, batch_size,
                                                                 n_abs_samples, medium, seed, self.constraint_kd_tree, self.get_poly_fit_options())

            print(f'Took {time.time() - t0} s')
            self.viewer_data = ViewerState(self, need_projection=False, poly_order=self.scatter_config.poly_order(),
                                           prediction_space='WS')
            self.coeffs = tmp_result['shapeCoeffs'][0]

            show_in_point_only = False
            if show_in_point_only:
                pos = vae.utils.mts_to_np(tmp_result['inPos'])[::batch_size, :]
                dirs = vae.utils.mts_to_np(tmp_result['inDir'])[::batch_size, :]
                coeffs = np.array(tmp_result['shapeCoeffs'])
                coeffs = coeffs[::batch_size, :]
                if self.importance_sample_train_data:
                    sigmoid = np.sum(coeffs[:, 4:] ** 2, axis=1)
                    sigmoid = np.log(sigmoid + 1e-4)
                    sigmoid = 1.0 / (1.0+np.exp(-sigmoid)) ** 4
                    criterion = sigmoid > np.random.rand(len(sigmoid))
                    colors = np.zeros((pos.shape[0], 3))
                    colors[:, 0] = sigmoid
                    self.viewer_data.append(pos, out_dirs=dirs, point_colors=colors)
                else:
                    self.viewer_data.append(pos, out_dirs=dirs)
            else:
                self.viewer_data.append(vae.utils.mts_to_np(tmp_result['outPos']),
                                        out_dirs=vae.utils.mts_to_np(tmp_result['outDir']))

    def keyboardEvent(self, key, scancode, action, modifiers):
        if super(Scatter3DViewer, self).keyboardEvent(key, scancode, action, modifiers):
            return True

        modes = [Mode.REF, Mode.PREDICTION, Mode.RECONSTRUCTION, Mode.POLYREF, Mode.POLYTRAIN]
        mode_keys = [glfw.KEY_1, glfw.KEY_2, glfw.KEY_3, glfw.KEY_4, glfw.KEY_5]

        def switch_mode(new_mode_index):
            self.mode_combobox.setSelectedIndex(new_mode_index)
            self.mode = modes[new_mode_index]
            self.update_displayed_scattering()
            self.performLayout()

        if action == glfw.PRESS and key in mode_keys:
            switch_mode(mode_keys.index(key))
            return True

        if action == glfw.PRESS and key == glfw.KEY_H:
            # Hide the UI
            visible = self.main_window.visible()
            self.main_window.setVisible(not visible)
            return True

        if action == glfw.PRESS and key == glfw.KEY_T:
            viewer.utils.dump_scattering_config(self, output_dir=os.path.join(
                vae.global_config.OUTPUT3D, 'figures', 'scatterpoints'))

        if action == glfw.PRESS and key == glfw.KEY_M:
            self.show_rec_mesh = not self.show_rec_mesh
            self.show_rec_mesh_checkbox.setChecked(self.show_rec_mesh)
            return True

        if action == glfw.PRESS and key == glfw.KEY_R:
            if self.recording_samples:
                for r in self.stored_samples:
                    r['inPos'] = vae.utils.mts_to_np(r['inPos'])
                    r['inDir'] = vae.utils.mts_to_np(r['inDir'])
                    r['outPos'] = vae.utils.mts_to_np(r['outPos'])
                    r['outDir'] = vae.utils.mts_to_np(r['outDir'])
                    r["inNormal"] = vae.utils.mts_to_np(r["inNormal"])
                    r["outNormal"] = vae.utils.mts_to_np(r["outNormal"])
                    r["albedo"] = vae.utils.mts_to_np(r["albedo"])
                    r["sigmaT"] = vae.utils.mts_to_np(r["sigmaT"])
                    r["throughput"] = vae.utils.mts_to_np(r["throughput"])
                with open(os.path.join(self.viewer_output_dir, 'samples.pickle'), 'wb') as f:
                    pickle.dump(self.stored_samples, f)
                self.stored_samples = []
                print('Saving recorded samples')
                self.recording_samples = False
            else:
                print('Start recording samples')
                self.recording_samples = True

        return False

    def mouseButtonEvent(self, p, button, action, modifier):
        if super(Scatter3DViewer, self).mouseButtonEvent(p, button, action, modifier):
            return True

        if button == 0 and action == glfw.PRESS and modifier == 0 and self.mesh is not None:
            self.its_loc, self.face_normal = intersect_mesh(p, self.camera, self.mesh)
            if self.its_loc is not None:
                self.picked_point.points = np.atleast_2d(self.its_loc.astype(np.float32)).T
                self.update_displayed_scattering()
                self.camera_controller.selectedPoint = self.its_loc

            return True

        if button == 1 and action == glfw.PRESS and modifier == 0 and self.mesh is not None:
            self.its_loc2, _ = intersect_mesh(p, self.camera, self.mesh)
            if self.its_loc2 is not None:
                self.picked_point2.points = np.atleast_2d(self.its_loc2.astype(np.float32)).T
                self.update_angular_histogram()
            return True
        return False

    def mouseMotionEvent(self, p, rel, button, modifier):
        if super().mouseMotionEvent(p, rel, button, modifier):
            return True

        if button == 2 and self.mesh is not None:
            self.its_loc2, _ = intersect_mesh(p, self.camera, self.mesh)
            if self.its_loc2 is not None:
                self.picked_point2.points = np.atleast_2d(self.its_loc2.astype(np.float32)).T
                self.update_angular_histogram()
            return True
        return False

    def drawContents(self):
        super().drawContents()
        self.render_context.shader.bind()
        self.render_context.its_loc = self.its_loc
        if self.mesh is not None:
            if not self.show_rec_mesh:
                if self.show_poly_coeffs:
                    if not self.computed_poly:
                        self.extract_mesh_polys()
                        self.computed_poly = True
                    n_vertices = self.mesh.mesh_positions.shape[1]
                    coeff = np.copy(self.mesh_polys[:, np.minimum(
                        self.poly_coeff_to_show, self.mesh_polys.shape[1] - 1)][:, None])
                    max_val = np.max(np.abs(coeff))
                    if self.poly_coeff_to_show > 0:
                        coeff /= max_val
                    red = np.ones((n_vertices, 3)) * np.array([1.0, 0.0, 0.0])
                    green = np.ones((n_vertices, 3)) * np.array([0.0, 1.0, 0.0])
                    colors = red * coeff
                    cond = (coeff <= 0).ravel()
                    colors[cond] = -green[cond] * coeff[cond]
                    self.mesh.draw_contents(self.camera, self.render_context, None, vertex_colors=colors)

                if self.show_fit_kernel and self.its_loc is not None:
                    coords = utils.math.grid_coordinates_3d(self.min_pos, self.max_pos, 64)
                    eps = vae.utils.kernel_epsilon(self.g, self.sigma_t, self.albedo, self.use_legacy_epsilon)
                    p = (self.its_loc.ravel())[None, None, None, :]
                    d2 = np.sum((coords - p) ** 2, -1)
                    w = np.exp(-d2 / (2 * eps))
                    self.mesh.draw_contents(self.camera, self.render_context, w,
                                            bb_min=self.min_pos, bb_max=self.max_pos)
                elif self.show_histograms:
                    if self.use_mesh_histogram:
                        self.viewer_data.get_mesh_histogram().draw_contents(self.camera, self.render_context)
                    else:
                        h = self.viewer_data.get_histogram()
                        self.mesh.draw_contents(self.camera, self.render_context, h,
                                                bb_min=self.min_pos, bb_max=self.max_pos)
                else:
                    self.mesh.draw_contents(self.camera, self.render_context, None)

        if self.training_points is not None:
            self.training_points.draw_contents(self.camera, self.render_context, None, [0, 1, 0])

        if self.inDirectionViz is not None:
            self.inDirectionViz.draw_contents(self.camera, self.render_context)

        if self.viewer_data is not None and self.show_rec_mesh:
            m = self.viewer_data.get_reconstructed_mesh()
            if m:
                m.draw_contents(self.camera, self.render_context, None, np.array([1, 0.8, 0.8, 1.0], np.float32))

        super().drawContentsFinalize()

    def drawContentsPost(self):
        if self.viewer_data is not None and not self.show_histograms and self.show_scatter_points:
            if self.project_samples or not self.viewer_data.need_projection:
                pts = self.viewer_data.points_pc
            else:
                pts = self.viewer_data.unproj_points_pc
            if pts is not None:
                if self.its_loc2 is not None:
                    pts.draw_contents(self.camera, self.render_context, None,
                                      disable_ztest=True, use_depth=True, depth_map=self.fb.depth(),
                                      use_ref_point=self.visualize_histogram_radius,
                                      ref_point=np.array(self.its_loc2), ref_radius=self.angular_histogram_radius,
                                      cull_occlusions=self.occlusion_culling)
                else:
                    pts.draw_contents(self.camera, self.render_context, None,
                                      disable_ztest=True, use_depth=True, depth_map=self.fb.depth(),
                                      cull_occlusions=self.occlusion_culling)

        if self.pos_constraints_pc is not None and self.show_used_constraints:
            self.pos_constraints_pc.draw_contents(self.camera, self.render_context, None, [
                                                  0, 1, 0], disable_ztest=True, use_depth=True, depth_map=self.fb.depth())

        if self.show_sampled_points:
            self.sampled_p.draw_contents(self.camera, self.render_context, None,
                                         [0, 1, 1], disable_ztest=True, use_depth=True, depth_map=self.fb.depth())
            self.sampled_n.draw_contents(self.camera, self.render_context)
        if self.show_projection_debug_info and self.viewer_data is not None and self.viewer_data.need_projection:
            self.viewer_data.unproj_points_pc.draw_contents(self.camera, self.render_context, None,
                                                            [0, 1, 0], disable_ztest=True, use_depth=True, depth_map=self.fb.depth())
            self.viewer_data.projection_vectors.draw_contents(self.camera, self.render_context)

        if self.show_outgoing_dir and self.viewer_data and self.viewer_data.out_dir_vectors:
            self.viewer_data.out_dir_vectors.draw_contents(self.camera, self.render_context)

        self.picked_point.draw_contents(self.camera, self.render_context, None,
                                        disable_ztest=True, use_depth=True, depth_map=self.fb.depth())

        if self.extra_point is not None:
            self.extra_point.draw_contents(self.camera, self.render_context, None,
                                           disable_ztest=True, use_depth=True, depth_map=self.fb.depth())

        if self.extra_dir is not None:
            self.extra_dir.draw_contents(self.camera, self.render_context)

        if self.show_angular_scattering:
            self.picked_point2.draw_contents(self.camera, self.render_context, None, color=[
                0, 0, 1], disable_ztest=True, use_depth=True, depth_map=self.fb.depth())


if __name__ == "__main__":
    nanogui.init()
    app = Scatter3DViewer(sys.argv[1:])
    app.drawAll()
    app.setVisible(True)
    nanogui.mainloop()
    print('Quitting...')
    # del app
    # gc.collect()
    nanogui.shutdown()
