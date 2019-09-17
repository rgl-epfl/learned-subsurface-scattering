import random

import numpy as np
import tensorflow as tf

import models.nn
from vae.config import *
import vae.config_scatter
import vae.utils


class BaselineAbs(vae.config_scatter.MlpShapeFeatures):
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

        # Handle outlier features: Filter in training data, clamp at testtime
        self.filter_feature_outliers = False
        self.abs_clamp_outlier_features = True
        self.abs_scaled_sigmoid_features = False
        self.sigmoid_scale_factor = 1.0

        self.n_abs_buckets = 8

        self.shape_feat_net = None

        self.learningrate = 0.0002


def generate_absorption_config():
    config = BaselineAbs()

    # Random number of layers
    config.n_abs_layers = np.random.randint(3, 8)

    # Random layer width
    if np.random.rand(1) > 0.7:
        # Use funnel architecture
        init_width = random.choice([32, 64, 128, 256])
        w = [init_width]
        for i in range(config.n_abs_layers - 1):
            w.append(w[-1] // 2)
        config.n_abs_width = w
    else:
        config.n_abs_width = random.choice([32, 48, 64, 96, 128, 256])

    # Batch size
    config.batch_size = random.choice([16, 32, 64, 128])

    # Use sigmoid feat
    config.abs_sigmoid_features = np.random.rand(1) > 0.4

    # Always clamp and filter outliers
    config.abs_clamp_outlier_features = True
    config.filter_feature_outliers = True

    # Add feat noise
    config.add_noise_to_poly_coeffs = np.random.rand(1) > 0.85

    # Add dropout
    if np.random.rand(1) > 0.99:
        config.dropout_keep_prob = 0.5 + 0.5 * np.random.rand(1)
    else:
        config.dropout_keep_prob = 1.0

    # Use a residual network instead
    if np.random.rand(1) > 0.7:
        config.abs_use_res_net = True
        # For now doesnt support funnel
        if isinstance(config.n_abs_width, list):
            config.n_abs_width = config.n_abs_width[0]
        config.use_batch_norm = True
    else:
        config.abs_use_res_net = False

    # Regularization
    if np.random.rand(1) > 0.5:
        reg_weight = 10 ** (-3*np.random.rand(1)-2)
        config.abs_regularizer = tf.contrib.layers.l2_regularizer(reg_weight)

    return config


class BaselineAbsDeg2(BaselineAbs):
    def __init__(self):
        super().__init__()
        self.shape_features_name = 'mlsPolyLS2'
        self.dataset = SCATTERDATADEG2


class BaselineAbsSim(BaselineAbs):
    def __init__(self):
        super().__init__()
        self.dataset = SCATTERDATASIMILARITY
        self.use_similarity_theory = True


class BaselineAbsSimSmall(BaselineAbsSim):
    def __init__(self):
        super().__init__()
        self.n_abs_layers = 4
        self.n_abs_width = 32


class BaselineAbsSmall(BaselineAbs):
    def __init__(self):
        super().__init__()
        self.n_abs_layers = 4
        self.n_abs_width = 32


class BaselineAbsShort(BaselineAbs):
    def __init__(self):
        super().__init__()
        self.n_abs_layers = 2


class BaselineAbsSimShort(BaselineAbsSim):
    def __init__(self):
        super().__init__()
        self.n_abs_layers = 2


class AbsSharedSim(BaselineAbsSim):  # Works best (makes the network more complex overall)
    def __init__(self):
        super().__init__()
        self.shape_feat_net = vae.model.shared_preproc_mlp
        self.n_abs_layers = 2
        self.n_abs_width = [32, 16]
        self.loss_weight = 5000


class AbsSharedSimComplex(BaselineAbsSim):  # Works best (makes the network more complex overall)
    def __init__(self):
        super().__init__()
        self.shape_feat_net = vae.model.shared_preproc_mlp_2
        self.n_abs_layers = 1
        self.n_abs_width = 32
        self.loss_weight = 5000


class AbsSharedSimComplexSlab(AbsSharedSimComplex):  # Works best (makes the network more complex overall)
    def __init__(self):
        super().__init__()
        self.dataset = SCATTERDATASLABS


class AbsSharedSimComplexMixed2(AbsSharedSimComplex):
    def __init__(self):
        super().__init__()
        self.dataset = SCATTERDATAMIXED2


class AbsSharedSimComplexMixed3(AbsSharedSimComplex):
    def __init__(self):
        super().__init__()
        self.dataset = SCATTERDATAMIXED3


class AbsSharedSimComplexMixed4(AbsSharedSimComplex):
    def __init__(self):
        super().__init__()
        self.dataset = SCATTERDATAMIXED4


class AbsSharedSimComplexAzimuth(AbsSharedSimComplex):
    def __init__(self):
        super().__init__()
        self.shape_features_name = 'mlsPolyAS3'
        self.polynomial_space = 'AS'
        self.prediction_space = 'AS'


class AbsSharedSimComplexAzimuthMixed2(AbsSharedSimComplexAzimuth):
    def __init__(self):
        super().__init__()
        self.dataset = SCATTERDATAMIXED2


class AbsSharedSimComplexAzimuthMixed3(AbsSharedSimComplexAzimuth):
    def __init__(self):
        super().__init__()
        self.dataset = SCATTERDATAMIXED3


class AbsSharedSimComplexAzimuthMixed4(AbsSharedSimComplexAzimuth):
    def __init__(self):
        super().__init__()
        self.dataset = SCATTERDATAMIXED4


class AbsFeatureSharedHigh(BaselineAbs):  # Best shared absorption predictor
    def __init__(self):
        super().__init__()
        self.shape_feat_net = vae.model.shared_preproc_mlp
        self.n_abs_layers = 1
        self.n_abs_width = 32
        self.loss_weight = 5000.0


class AbsFeatureShared2(BaselineAbs):  # Works best (makes the network more complex overall)
    def __init__(self):
        super().__init__()
        self.shape_feat_net = vae.model.shared_preproc_mlp
        self.n_abs_layers = 2
        self.n_abs_width = 64
        self.loss_weight = 100.0


class AbsFeatureShared3(BaselineAbs):  # Best shared absorption predictor
    def __init__(self):
        super().__init__()
        self.shape_feat_net = vae.model.shared_preproc_mlp
        self.n_abs_layers = 1
        self.n_abs_width = 64
        self.loss_weight = 400.0


class AbsFeatureShared4(BaselineAbs):  # Works best (makes the network more complex overall)
    def __init__(self):
        super().__init__()
        self.shape_feat_net = vae.model.shared_preproc_mlp
        self.n_abs_layers = 1
        self.n_abs_width = 64
        self.loss_weight = 50.0


class AbsFeatureShared5(BaselineAbs):  # Works best (makes the network more complex overall)
    def __init__(self):
        super().__init__()
        self.shape_feat_net = vae.model.shared_preproc_mlp
        self.n_abs_layers = 1
        self.n_abs_width = 32
        self.loss_weight = 10.0


class AbsFeatureShared6(BaselineAbs):  # Works best (makes the network more complex overall)
    def __init__(self):
        super().__init__()
        self.shape_feat_net = vae.model.shared_preproc_mlp
        self.n_abs_layers = 1
        self.n_abs_width = 32
        self.loss_weight = 20.0


class BaselineAbsHardConstraint(BaselineAbs):
    def __init__(self):
        super().__init__()
        self.dataset = SCATTERDATASURFACECONSTRAINT


class AbsShCoefficients(BaselineAbs):
    def __init__(self):
        super().__init__()
        self.dataset = SCATTERDATASH
        self.use_sh_coeffs = True
        self.shape_features_name = 'shCoeffs'
        self.pass_in_dir = True


class AbsRotatedFeatures(BaselineAbs):
    def __init__(self):
        super().__init__()
        self.shape_features_name = 'mlsPolyTS3'
        self.polynomial_space = 'TS'
        self.rotate_features = True
        self.pass_in_dir = False

        self.shape_feat_net = lambda x, is_training: models.nn.multilayer_fcn(
            x, is_training, None, 3, 32, self.use_batch_norm, 'shapemlp', n_out=36)


class AbsRotatedFeatures2(AbsRotatedFeatures):
    def __init__(self):
        super().__init__()
        self.first_layer_feats = True
        self.first_layer_feats = True
        self.n_mlp_layers = 2
        self.n_mlp_width = 32
        self.shape_feat_net = lambda x, is_training: models.nn.multilayer_fcn(
            x, is_training, None, 3, 64, self.use_batch_norm, 'shapemlp', n_out=48)


class AbsRotatedFeatures3(AbsRotatedFeatures):
    def __init__(self):
        super().__init__()
        self.first_layer_feats = True
        self.first_layer_feats = True
        self.n_mlp_layers = 1
        self.n_mlp_width = 32
        self.shape_feat_net = lambda x, is_training: models.nn.multilayer_fcn(
            x, is_training, None, 3, 64, self.use_batch_norm, 'shapemlp', n_out=48)


# class AbsShCoefficientsRotated(AbsShCoefficients):
#     def __init__(self):
#         super().__init__()
#         self.rotate_features = True


class AbsShCoeffsShared(AbsShCoefficients):
    def __init__(self):
        super().__init__()
        self.shape_feat_net = vae.model.shared_preproc_mlp
        self.n_abs_layers = 3
        self.n_abs_width = 32


# class AbsShCoeffsSharedRotated(AbsShCoefficientsRotated):
#     def __init__(self):
#         super().__init__()
#         self.shape_feat_net = vae.model.shared_preproc_mlp
#         self.n_abs_layers = 3
#         self.n_abs_width = 32


class AbsPointnet(BaselineAbs):
    def __init__(self):
        super().__init__()
        self.use_point_net = True
        self.n_point_net_points = 64
        self.dataset = '0062_ScatterDataPc'
        self.shape_features_name = None


class AbsPointNetNormals(AbsPointnet):
    def __init__(self):
        super().__init__()
        self.point_net_use_normals = True


class AbsPointNetWeighted(AbsPointnet):
    def __init__(self):
        super().__init__()
        self.point_net_use_weights = True


class AbsPointnetTrue(BaselineAbs):
    def __init__(self):
        super().__init__()
        self.use_point_net = True
        self.n_point_net_points = 64
        self.dataset = SCATTERDATATRUE
        self.shape_features_name = None
        self.point_net_use_normals = True
        self.point_net_use_weights = False


class AbsPointnetShared(AbsPointnetTrue):
    def __init__(self):
        super().__init__()
        self.point_net = vae.model.baseline_point_net_shared


class AbsPointnetHist(AbsPointnetTrue):
    def __init__(self):
        super().__init__()
        self.point_net_normal_histogram = True
        self.point_net_n_normal_bins = 3
        self.point_net_feature_sizes = [8, 16, 16]
        self.point_net_use_weights = False


class AbsPointnetHist5(AbsPointnetHist):
    def __init__(self):
        super().__init__()
        self.point_net_n_normal_bins = 5


class AbsLr1(BaselineAbs):
    def __init__(self):
        super().__init__()
        self.learningrate = 0.001


class AbsLr2(BaselineAbs):
    def __init__(self):
        super().__init__()
        self.learningrate = 0.01


class AbsLr3(BaselineAbs):
    def __init__(self):
        super().__init__()
        self.learningrate = 0.0001


class AbsLr4(BaselineAbs):
    def __init__(self):
        super().__init__()
        self.learningrate = 0.00005


class AbsLegendre(BaselineAbs):
    def __init__(self):
        super().__init__()
        self.shape_features_name = 'legendrePolyLS3'


class AbsLegendreDeg2(BaselineAbs):
    def __init__(self):
        super().__init__()
        self.shape_features_name = 'legendrePolyLS2'
        self.dataset = SCATTERDATADEG2


class AbsNoKdThreshold(BaselineAbs):
    def __init__(self):
        super().__init__()
        self.dataset = '0063_ScatterDataNoThreshold'


class AbsMoreDataReg(BaselineAbs):
    def __init__(self):
        super().__init__()
        self.dataset = '0064_ScatterDataMoreReg'


class AbsNoKdThresholdDeg2(BaselineAbsDeg2):
    def __init__(self):
        super().__init__()
        self.dataset = '0065_ScatterDataNoThresholdDeg2'


class AbsClassification1(BaselineAbs):
    def __init__(self):
        super().__init__()
        self.abs_loss = 'classification'
        self.n_abs_buckets = 8


class AbsClassification2(AbsClassification1):
    def __init__(self):
        super().__init__()
        self.n_abs_buckets = 16


class AbsEvalPoly(BaselineAbs):
    def __init__(self):
        super().__init__()
        self.eval_poly_feature = True


class AbsEvalPolySigmoid(BaselineAbs):
    def __init__(self):
        super().__init__()
        self.eval_poly_feature = True
        self.sigmoid_features = True


class AbsEvalPolyBinary(BaselineAbs):
    def __init__(self):
        super().__init__()
        self.binary_features = True
        self.eval_poly_feature = True


class AbsEvalPolyCnn(BaselineAbs):
    def __init__(self):
        super().__init__()
        self.binary_features = False
        self.eval_poly_feature = True

        self.poly_eval_range = 2.0
        self.poly_eval_steps = 8
        self.poly_conv_net = vae.model.polyCnn


class AbsEvalPolyBinaryCnn(AbsEvalPolyCnn):
    def __init__(self):
        super().__init__()
        self.binary_features = True


class AbsEvalPolySigmoidCnn(AbsEvalPolyCnn):
    def __init__(self):
        super().__init__()
        self.sigmoid_features = True


class AbsFeaturePre(BaselineAbs):  # Works best (makes the network more complex overall)
    def __init__(self):
        super().__init__()
        self.shape_feat_net = lambda x, is_training: models.nn.multilayer_fcn(
            x, is_training, None, 3, 32, self.use_batch_norm, 'shapemlp')
        self.n_abs_layers = 1
        self.n_abs_width = 32


class AbsFeatureShared(BaselineAbs):  # Works best (makes the network more complex overall)
    def __init__(self):
        super().__init__()
        self.shape_feat_net = vae.model.shared_preproc_mlp
        self.n_abs_layers = 1
        self.n_abs_width = 32


class AbsFeatureSharedComplex(BaselineAbs):  # Works best (makes the network more complex overall)
    def __init__(self):
        super().__init__()
        self.shape_feat_net = vae.model.shared_preproc_mlp_2
        self.n_abs_layers = 1
        self.n_abs_width = 32
        self.loss_weight = 10.0


class AbsFeatureSharedDeg2(BaselineAbsDeg2):  # Works best (makes the network more complex overall)
    def __init__(self):
        super().__init__()
        self.shape_feat_net = vae.model.shared_preproc_mlp
        self.n_abs_layers = 1
        self.n_abs_width = 32


class AbsSigmoidFlt(BaselineAbs):
    def __init__(self):
        super().__init__()
        self.filter_feature_outliers = True
        self.abs_sigmoid_features = True
        self.n_abs_layers = 5
        self.n_abs_width = 32
        self.batch_size = 16


class AbsClampFeatures(BaselineAbs):
    def __init__(self):
        super().__init__()
        self.abs_clamp_outlier_features = True


class AbsClampFeaturesSigmoid(BaselineAbs):
    def __init__(self):
        super().__init__()
        self.abs_clamp_outlier_features = True
        self.abs_sigmoid_features = True


class AbsSigmoid(BaselineAbs):
    def __init__(self):
        super().__init__()
        self.abs_sigmoid_features = True


class AbsSigmoidScaled(BaselineAbs):
    def __init__(self):
        super().__init__()
        self.abs_scaled_sigmoid_features = True
        self.sigmoid_scale_factor = 0.2
        self.filter_feature_outliers = False
        self.abs_clamp_outlier_features = False


class AbsSigmoidScaled01(AbsSigmoidScaled):
    def __init__(self):
        super().__init__()
        self.sigmoid_scale_factor = 0.1


class AbsSigmoidScaled05(AbsSigmoidScaled):
    def __init__(self):
        super().__init__()
        self.sigmoid_scale_factor = 0.5


class AbsD4(BaselineAbs):
    def __init__(self):
        super().__init__()
        self.n_abs_layers = 4


class AbsD5(BaselineAbs):
    def __init__(self):
        super().__init__()
        self.n_abs_layers = 5


class AbsD6(BaselineAbs):
    def __init__(self):
        super().__init__()
        self.n_abs_layers = 6


class AbsD7(BaselineAbs):
    def __init__(self):
        super().__init__()
        self.n_abs_layers = 7


class AbsW64(BaselineAbs):
    def __init__(self):
        super().__init__()
        self.n_abs_width = 64


class AbsW128(BaselineAbs):
    def __init__(self):
        super().__init__()
        self.n_abs_width = 128
        self.abs_regularizer = tf.contrib.layers.l2_regularizer(0.00001)


class AbsSigmoidReg(AbsSigmoidFlt):
    def __init__(self):
        super().__init__()
        self.abs_regularizer = tf.contrib.layers.l2_regularizer(0.0001)


class AbsGammaWithAlbedo(BaselineAbs):
    def __init__(self):
        super().__init__()
        self.abs_bounce_distribution = 'gamma'


class AbsGammaWithoutAlbedo(BaselineAbs):
    def __init__(self):
        super().__init__()
        self.abs_bounce_distribution = 'gamma'
        self.abs_use_albedo_feature = False


class AbsFunnel(BaselineAbs):
    def __init__(self):
        super().__init__()
        self.n_abs_layers = 3
        self.n_abs_width = [64, 32, 8]


class AbsFunnel2(BaselineAbs):
    def __init__(self):
        super().__init__()
        self.n_abs_layers = 4
        self.n_abs_width = [64, 32, 8, 2]


class AbsFunnel3(BaselineAbs):
    def __init__(self):
        super().__init__()
        self.n_abs_layers = 4
        self.n_abs_width = [32, 64, 32, 16]


class AbsFunnel3Reg(AbsFunnel3):
    def __init__(self):
        super().__init__()
        self.abs_regularizer = tf.contrib.layers.l2_regularizer(0.001)


class AbsFunnel4(BaselineAbs):
    def __init__(self):
        super().__init__()
        self.n_abs_layers = 4
        self.n_abs_width = [64, 32, 16, 8, 4, 2]


class AbsFunnel5(BaselineAbs):
    def __init__(self):
        super().__init__()
        self.n_abs_layers = 6
        self.n_abs_width = [64, 32, 16, 8, 4, 2]


class AbsFunnel6(BaselineAbs):
    def __init__(self):
        super().__init__()
        self.n_abs_layers = 5
        self.n_abs_width = [32, 64, 32, 16, 8]


class AbsorptionDeep(BaselineAbs):
    def __init__(self):
        super().__init__()
        self.n_abs_layers = 5


class AbsorptionW64(BaselineAbs):
    def __init__(self):
        super().__init__()
        self.n_abs_width = 64


class AbsorptionW64Reg(AbsorptionW64):

    def __init__(self):
        super().__init__()
        self.abs_regularizer = tf.contrib.layers.l2_regularizer(0.001)


class AbsorptionReg(BaselineAbs):

    def __init__(self):
        super().__init__()
        self.abs_regularizer = tf.contrib.layers.l2_regularizer(0.001)


class AbsorptionReg2(BaselineAbs):
    def __init__(self):
        super().__init__()
        self.abs_regularizer = tf.contrib.layers.l2_regularizer(0.01)


class AbsorptionReg3(BaselineAbs):
    def __init__(self):
        super().__init__()
        self.abs_regularizer = tf.contrib.layers.l2_regularizer(0.1)


class AbsorptionReg4(BaselineAbs):
    def __init__(self):
        super().__init__()
        self.abs_regularizer = tf.contrib.layers.l2_regularizer(0.0001)


class AbsorptionWide(BaselineAbs):
    def __init__(self):
        super().__init__()
        self.n_abs_width = 128


class AbsorptionWideReg(AbsorptionWide):
    def __init__(self):
        super().__init__()
        self.abs_regularizer = tf.contrib.layers.l2_regularizer(0.001)


class AbsorptionWideReg2(AbsorptionWide):
    def __init__(self):
        super().__init__()
        self.abs_regularizer = tf.contrib.layers.l2_regularizer(0.0001)


class AbsorptionResnet(BaselineAbs):
    def __init__(self):
        super().__init__()
        self.abs_use_res_net = True


class AbsorptionResnet2(BaselineAbs):
    def __init__(self):
        super().__init__()
        self.abs_use_res_net = True
        self.n_abs_width = 64


class AbsorptionResnet3(BaselineAbs):
    def __init__(self):
        super().__init__()
        self.abs_use_res_net = True
        self.n_abs_width = 64
        self.n_abs_layers = 5
        self.abs_regularizer = tf.contrib.layers.l2_regularizer(0.0001)
        self.abs_sigmoid_features = True


class AbsorptionResnet4(BaselineAbs):
    def __init__(self):
        super().__init__()
        self.abs_use_res_net = True
        self.n_abs_width = 64
        self.n_abs_layers = 6
        self.abs_regularizer = tf.contrib.layers.l2_regularizer(0.00001)


class AbsorptionResnet5(BaselineAbs):
    def __init__(self):
        super().__init__()
        self.abs_use_res_net = True
        self.n_abs_width = 64
        self.n_abs_layers = 6
        self.abs_regularizer = tf.contrib.layers.l2_regularizer(0.00001)
        self.abs_sigmoid_features = True


class AbsorptionDeeper(BaselineAbs):
    def __init__(self):
        super().__init__()
        self.n_abs_layers = 7


class AbsorptionDeeperReg(AbsorptionDeeper):
    def __init__(self):
        super().__init__()
        self.abs_regularizer = tf.contrib.layers.l2_regularizer(0.001)


class AbsorptionCrossEntropy(BaselineAbs):
    def __init__(self):
        super().__init__()
        self.abs_loss = 'crossentropy'


class AbsorptionL1(BaselineAbs):
    def __init__(self):
        super().__init__()
        self.abs_loss = 'l1'


class AbsorptionDropout05(BaselineAbs):
    def __init__(self):
        super().__init__()
        self.dropout_keep_prob = 0.5
        self.absorption_model = vae.model.absorptionMlpDropout


class AbsorptionDropout07(BaselineAbs):
    def __init__(self):
        super().__init__()
        self.dropout_keep_prob = 0.7
        self.absorption_model = vae.model.absorptionMlpDropout


class AbsorptionDropout09(BaselineAbs):
    def __init__(self):
        super().__init__()
        self.dropout_keep_prob = 0.9
        self.absorption_model = vae.model.absorptionMlpDropout


vae.utils.add_variants(globals(), [AbsFunnel, AbsFunnel2, AbsFunnel3, AbsFunnel4, AbsorptionDeep,
                                   AbsorptionWide, AbsorptionResnet, AbsorptionDeeper, AbsGammaWithAlbedo, AbsorptionDeeper, AbsorptionResnet2, AbsorptionResnet3,
                                   AbsorptionWideReg, AbsorptionWideReg2, AbsorptionDropout05, AbsorptionDropout07, AbsorptionDropout09, AbsFunnel3Reg,
                                   AbsorptionW64, AbsorptionW64Reg],
                       {'seed': 51}, 'Seed')


class AbsorptionSimple(BaselineAbs):
    def __init__(self):
        super().__init__()
        self.absorption_model = AbsorptionSimple.absorptionSimple

    @staticmethod
    def absorptionSimple(x, trainer, config):
        x = tf.concat([x, tf.square(x)])
        return models.nn.multilayer_fcn(x, trainer.phase_p, None, 1, 32, False, 'mlp')


class AbsorptionSigmoid(BaselineAbs):
    def __init__(self):
        super().__init__()
        self.absorption_model = AbsorptionSigmoid.absorptionSigmoid

    @staticmethod
    def absorptionSigmoid(x, trainer, config):
        x = tf.nn.sigmoid(x)
        return models.nn.multilayer_fcn(x, trainer.phase_p, None,
                                        config.n_abs_layers, config.n_abs_width, False, 'mlp')


class AbsorptionFeatNoise(BaselineAbs):
    def __init__(self):
        super().__init__()
        self.add_noise_to_poly_coeffs = True


class AbsorptionSFeatNoise(AbsorptionSigmoid):
    def __init__(self):
        super().__init__()
        self.add_noise_to_poly_coeffs = True


class AbsorptionSSmallLR(AbsorptionSigmoid):
    def __init__(self):
        super().__init__()
        self.learningrate = 0.0001


class AbsorptionSSmallLRSeed(AbsorptionSSmallLR):
    def __init__(self):
        super().__init__()
        self.seed = 51


class AbsorptionSConcat(BaselineAbs):
    def __init__(self):
        super().__init__()
        self.absorption_model = AbsorptionSConcat.absorptionSigmoidConcat

    @staticmethod
    def absorptionSigmoidConcat(x, trainer, config):
        x = tf.nn.sigmoid(x)
        return models.nn.multilayer_fcn(x, trainer.phase_p, x,
                                        config.n_abs_layers, config.n_abs_width, False, 'mlp')


class AbsorptionSBatch64(AbsorptionSigmoid):
    def __init__(self):
        super().__init__()
        self.batch_size = 64


class AbsorptionSBatch64Deep(AbsorptionSigmoid):
    def __init__(self):
        super().__init__()
        self.batch_size = 64
        self.n_abs_layers = 5


class AbsorptionSBatch128(AbsorptionSigmoid):
    def __init__(self):
        super().__init__()
        self.batch_size = 128


class AbsorptionSBatch128Seed(AbsorptionSigmoid):
    def __init__(self):
        super().__init__()
        self.batch_size = 128
        self.seed = 51


class AbsorptionBatch128(BaselineAbs):
    def __init__(self):
        super().__init__()
        self.batch_size = 128


class AbsorptionBatch128Seed(BaselineAbs):
    def __init__(self):
        super().__init__()
        self.batch_size = 128
        self.seed = 51


class AbsorptionSDropout05(AbsorptionSigmoid):
    def __init__(self):
        super().__init__()
        self.dropout_keep_prob = 0.5
        self.absorption_model = vae.model.absorptionMlpDropout

    @staticmethod
    def absorptionMlpDropout(x, trainer, config):
        x = tf.nn.sigmoid(x)
        return models.nn.multilayer_fcn(x, trainer.phase_p, None,
                                        config.n_abs_layers, config.n_abs_width, False, 'mlp',
                                        use_dropout=True, dropout_keep_prob=trainer.dropout_keep_prob_p)


class AbsorptionSDropout07(AbsorptionSDropout05):
    def __init__(self):
        super().__init__()
        self.dropout_keep_prob = 0.7


class AbsorptionSDropout09(AbsorptionSDropout05):
    def __init__(self):
        super().__init__()
        self.dropout_keep_prob = 0.9


class AbsorptionSDropout05Deep(AbsorptionSDropout05):
    def __init__(self):
        super().__init__()
        self.n_abs_layers = 5


class AbsorptionMlpConcat(BaselineAbs):
    def __init__(self):
        super().__init__()
        self.absorption_model = AbsorptionMlpConcat.absorptionMlpConcat

    @staticmethod
    def absorptionMlpConcat(x, trainer, config):
        return models.nn.multilayer_fcn(x, trainer.phase_p, x,
                                        config.n_abs_layers, config.n_abs_width, False, 'mlp')


class AbsFixedMedium(BaselineAbs):
    def __init__(self):
        super().__init__()
        self.dataset = vae.datapipeline.ScatterDataFixedMedium


class AbsFixedMediumSphere(BaselineAbs):
    def __init__(self):
        super().__init__()
        self.dataset = vae.datapipeline.ScatterDataSphereFixedMedium
