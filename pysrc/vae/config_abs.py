import random

import numpy as np
import tensorflow as tf

import models.nn
from vae.config import *
import vae.config_scatter
import vae.utils


class AbsorptionModel(ConfigBase):

    def __init__(self):
        super().__init__()
        self.num_epochs = 5000
        self.abs_bounce_distribution = None
        self.model = vae.model.absorptionPredictor
        self.abs_predict_diff = False
        self.abs_loss = 'l2'
        self.n_abs_layers = 3
        self.n_abs_width = 64
        self.absorption_model = vae.model.absorptionMlp
        self.abs_bounce_distribution = None
        self.abs_use_albedo_feature = True
        self.abs_regularizer = None

        self.abs_clamp_outlier_features = True
        self.abs_scaled_sigmoid_features = False
        self.sigmoid_scale_factor = 1.0
        self.n_abs_buckets = 8

        self.shape_feat_net = None
        self.learningrate = 0.0002

        self.use_similarity_theory = True

        self.shape_feat_net = vae.model.shared_preproc_mlp_2
        self.n_abs_layers = 1
        self.n_abs_width = 32
        self.loss_weight = 5000