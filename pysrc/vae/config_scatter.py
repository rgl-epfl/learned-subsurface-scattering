import tensorflow as tf

from vae.config import *

import models.nn

# Current best model

class VaeScatter(ConfigBase):

    def __init__(self):
        super().__init__()
        self.model = vae.model.baselineShapeDescriptor
        self.convnet = None
        self.use_wae_mmd = False
        self.wae_random_enc = False
        self.add_encoder_noise = False  # Adds noise to the input of the encoder to reduce overfitting

        self.use_cnn_features = False
        self.use_res_net = False
        self.use_batch_norm = False
        self.n_mlp_layers = 3
        self.n_mlp_width = 64
        self.predict_in_tangent_space = True

        self.filter_feature_outliers = False

        self.use_coupling_coord_offset = False

        self.use_similarity_theory = True

        self.n_latent = 4
        self.first_layer_feats = True
        self.shape_feat_net = vae.model.shared_preproc_mlp_2

        self.filter_outliers = False
        self.use_epsilon_space = True
        self.gen_loss_weight = 10000
        self.use_outpos_statistics = False

        self.shape_features_name = 'mlsPolyLS3'
        self.polynomial_space = 'LS'
        self.prediction_space = 'LS'

        self.gen_loss = 'l2'
        self.clip_gradient_threshold = 100
