import numpy as np
import tensorflow as tf

import models.nn
import vae.config
from vae.config import *
import vae.config_scatter
import vae.utils
import vae.model


class BaselineAngular(vae.config_scatter.MlpShapeFeatures):
    def __init__(self):
        super().__init__()
        self.model = vae.model.baselineAngularScatter
        self.use_nice = True
        self.n_latent = 2
        self.batch_size = 128
        self.learningrate = 2e-4

        self.clip_gradient_threshold = 100
        self.filter_outliers = True

        self.use_res_net = True
        self.use_batch_norm = False

        self.n_coupling_layers = 4

        # Handle outlier features: Filter in training data, clamp at testtime
        self.filter_feature_outliers = True
        self.abs_clamp_outlier_features = True

        self.use_vmf = False




class DebugAngular(BaselineAngular):
    def __init__(self):
        super().__init__()
        self.model = vae.model.debugAngularScatter


class VmfAngular(BaselineAngular):
    def __init__(self):
        super().__init__()

        self.use_vmf = True
        self.model = vae.model.vmfAngularScatter

        self.n_mlp_layers = 3
        self.n_mlp_width = [64, 32, 16]


class VmfAngularPre(VmfAngular):  # Works best (makes the network more complex overall)
    def __init__(self):
        super().__init__()
        self.shape_feat_net = lambda x, is_training: models.nn.multilayer_fcn(
            x, is_training, None, 3, 32, self.use_batch_norm, 'shapemlp')
        self.n_mlp_layers = 2
        self.n_mlp_width = [32, 16]


class VmfAngularPreShared(VmfAngular):  # Works best (makes the network more complex overall)
    def __init__(self):
        super().__init__()
        self.shape_feat_net = vae.model.shared_preproc_mlp
        self.n_mlp_layers = 2
        self.n_mlp_width = [32, 16]


class VmfFixedMedium(VmfAngular):
    def __init__(self):
        super().__init__()
        self.dataset = vae.datapipeline.ScatterDataFixedMedium
        self.num_epochs = 100


class VmfFixedMediumSphere(VmfAngular):
    def __init__(self):
        super().__init__()
        self.dataset = SCATTERDATASPHEREFIXEDMEDIUM
        self.num_epochs = 100


class VmfFixedMediumSphereSearchlightDeg2(VmfAngular):
    def __init__(self):
        super().__init__()
        self.shape_features_name = 'mlsPolyLS2'
        self.dataset = SCATTERDATASPHEREFIXEDMEDIUMSEARCHLIGHTDEG2
        self.num_epochs = 100


class VmfPlaneFixedMediumSearchlight(VmfAngular):
    def __init__(self):
        super().__init__()
        self.shape_features_name = 'mlsPolyLS1'
        self.dataset = SCATTERDATAPLANEFIXEDMEDIUMSEARCHLIGHT
        self.num_epochs = 100
