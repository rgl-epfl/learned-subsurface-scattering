import tensorflow as tf

from vae.config import *

import models.nn

# Current best model


class MlpShapeFeatures(ConfigBase):
    def __init__(self):
        super().__init__()
        self.model = vae.model.baselineShapeDescriptor
        self.convnet = None
        self.use_wae_mmd = False
        self.wae_random_enc = False
        self.add_encoder_noise = False  # Adds noise to the input of the encoder to reduce overfitting

        self.use_cnn_features = False
        self.n_latent = 8
        self.gen_loss_weight = 1000
        self.use_res_net = False
        self.use_batch_norm = False
        self.n_mlp_layers = 3
        self.n_mlp_width = 64
        self.shape_features_name = 'mlsPolyLS3'
        self.predict_in_tangent_space = True
        self.prediction_space = 'LS'
        self.polynomial_space = 'LS'

        # Seems to be more robust than L2
        self.gen_loss = 'huber'

        # Additionally clip gradient and filter outliers
        self.clip_gradient_threshold = 100
        self.filter_outliers = True
        self.filter_feature_outliers = False

        self.use_coupling_coord_offset = False
        self.first_layer_feats = False

        self.shape_feat_net = lambda x, is_training: models.nn.multilayer_fcn(
            x, is_training, None, 3, 32, self.use_batch_norm, 'shapemlp')

    def poly_order(self):
        if not self.shape_features_name:
            return 0
        return vae.utils.extract_poly_order_from_feat_name(self.shape_features_name)


class MlpShapeFeaturesDeg2(MlpShapeFeatures):
    def __init__(self):
        super().__init__()
        self.dataset = SCATTERDATADEG2
        self.shape_features_name = 'mlsPolyLS2'

class MlpShapeFeaturesSphere(MlpShapeFeatures):
    def __init__(self):
        super().__init__()
        self.dataset = SCATTERDATASPHEREFIXEDMEDIUMSEARCHLIGHT

class MlpOldDataset(MlpShapeFeatures):
    def __init__(self):
        super().__init__()
        self.dataset = SCATTERDATASIMILARITYOLD



class VaeEfficient(MlpShapeFeatures):
    def __init__(self):
        super().__init__()
        self.n_latent = 4
        self.first_layer_feats = True


class VaeEfficientSim(VaeEfficient):
    def __init__(self):
        super().__init__()
        self.dataset = SCATTERDATASIMILARITY
        self.use_similarity_theory = True


class MlpShapeFeaturesSim(MlpShapeFeatures):
    def __init__(self):
        super().__init__()
        self.dataset = SCATTERDATASIMILARITY
        self.use_similarity_theory = True


class MlpShapeFeaturesSimG100(MlpShapeFeaturesSim):
    def __init__(self):
        super().__init__()
        self.gen_loss_weight = 100


class MlpShapeFeaturesSimG200(MlpShapeFeaturesSim):
    def __init__(self):
        super().__init__()
        self.gen_loss_weight = 200


class MlpShapeFeaturesSimG500(MlpShapeFeaturesSim):
    def __init__(self):
        super().__init__()
        self.gen_loss_weight = 500


class MlpShapeFeaturesSimG750(MlpShapeFeaturesSim):
    def __init__(self):
        super().__init__()
        self.gen_loss_weight = 750

class MlpShapeFeaturesSimG50(MlpShapeFeaturesSim):
    def __init__(self):
        super().__init__()
        self.gen_loss_weight = 50


class VaeEfficientSimScaled(VaeEfficientSim):
    def __init__(self):
        super().__init__()
        self.scale_point_by_poly_scale = True


class MlpShapeFeaturesSimScaled(MlpShapeFeaturesSim):
    def __init__(self):
        super().__init__()
        self.scale_point_by_poly_scale = True



class VaeFeaturePreSharedSim(MlpShapeFeaturesSim):  # Works best (makes the network more complex overall)
    def __init__(self):
        super().__init__()
        self.n_latent = 4
        self.first_layer_feats = True
        self.shape_feat_net = vae.model.shared_preproc_mlp

class SmallVaeFeaturePreSharedSimSkip(VaeFeaturePreSharedSim):  # Works best (makes the network more complex overall)
    def __init__(self):
        super().__init__()
        self.first_layer_feats = False

class VaeFeaturePreSharedSim2(MlpShapeFeaturesSim):  # BEST FINAL NETWORK
    def __init__(self):
        super().__init__()
        self.n_latent = 4
        self.first_layer_feats = True
        self.shape_feat_net = vae.model.shared_preproc_mlp_2

# FinalSharedAz/AbsSharedSimComplexAzimuth

class FinalSharedAz(VaeFeaturePreSharedSim2): # Train with /AbsSharedSimComplex
    def __init__(self):
        super().__init__()
        self.filter_outliers = False
        self.use_epsilon_space = True
        self.gen_loss_weight = 10000
        self.use_outpos_statistics = False
        self.shape_features_name = 'mlsPolyAS3'
        self.polynomial_space = 'AS'
        self.prediction_space = 'AS'

class FinalSharedAz2(FinalSharedAz):
    def __init__(self):
        super().__init__()
        self.gen_loss_weight = 5000

class FinalSharedAz3(FinalSharedAz):
    def __init__(self):
        super().__init__()
        self.gen_loss_weight = 2000

class FinalSharedLs(FinalSharedAz):
    def __init__(self):
        super().__init__()
        self.shape_features_name = 'mlsPolyLS3'
        self.polynomial_space = 'LS'
        self.prediction_space = 'LS'

class SlabsSharedLs(FinalSharedLs):
    def __init__(self):
        super().__init__()
        self.dataset = SCATTERDATASLABS

class SlabsSharedLs2(VaeFeaturePreSharedSim2):
    def __init__(self):
        super().__init__()
        self.dataset = SCATTERDATASLABS

class FinalSharedLs2(FinalSharedLs): # Bad
    def __init__(self):
        super().__init__()
        self.gen_loss_weight = 2000

class FinalSharedLs3(FinalSharedLs):
    def __init__(self):
        super().__init__()
        self.gen_loss_weight = 5000


class FinalSharedLs4(FinalSharedLs):
    def __init__(self):
        super().__init__()
        self.gen_loss_weight = 20000

class FinalSharedLs5(FinalSharedLs): # Bad, scattering almost non existent
    def __init__(self):
        super().__init__()
        self.gen_loss_weight = 500

class FinalSharedLs6(FinalSharedLs): # Very good: Maybe the outlier filtering is problematic?
    def __init__(self):
        super().__init__()
        self.filter_outliers = False

class FinalSharedLs7(FinalSharedLs): # BEST RESULT 19/4/2019
    def __init__(self):
        super().__init__()
        self.filter_outliers = False
        self.gen_loss = 'l2' # Using l2 loss seems to be bad as it gives outliers
        self.clip_gradient_threshold = 100

class FinalSharedLsHuber(FinalSharedLs7):
    def __init__(self):
        super().__init__()
        self.gen_loss = 'huber'

class FinalSharedLs1000(FinalSharedLs7):
    def __init__(self):
        super().__init__()
        self.gen_loss_weight = 1000

class FinalSharedLs3d(FinalSharedLs7):
    def __init__(self):
        super().__init__()
        self.n_latent = 3
        

class FinalSharedLs7Mixed2(FinalSharedLs7):
    def __init__(self):
        super().__init__()
        self.dataset = '0117_ScatterDataMixed2'

class FinalSharedLs7Mixed3(FinalSharedLs7):
    def __init__(self):
        super().__init__()
        self.dataset = '0118_ScatterDataMixed3'

class FinalSharedLs7Mixed4(FinalSharedLs7):
    def __init__(self):
        super().__init__()
        self.dataset = '0119_ScatterDataMixed4'


class FinalSharedLs6Mixed2(FinalSharedLs6):
    def __init__(self):
        super().__init__()
        self.dataset = SCATTERDATAMIXED2

class FinalSharedLs6Mixed3(FinalSharedLs6):
    def __init__(self):
        super().__init__()
        self.dataset = SCATTERDATAMIXED3

class FinalSharedLs6Mixed4(FinalSharedLs6):
    def __init__(self):
        super().__init__()
        self.dataset = SCATTERDATAMIXED4


class FinalSharedAzMixed2(FinalSharedAz):
    def __init__(self):
        super().__init__()
        self.dataset = SCATTERDATAMIXED2

class FinalSharedAzMixed3(FinalSharedAz):
    def __init__(self):
        super().__init__()
        self.dataset = SCATTERDATAMIXED3

class FinalSharedAzMixed4(FinalSharedAz):
    def __init__(self):
        super().__init__()
        self.dataset = SCATTERDATAMIXED4



class FinalSharedLs8(FinalSharedLs): # not very good
    def __init__(self):
        super().__init__()
        self.filter_outliers = False
        self.gen_loss = 'l2'
        self.clip_gradient_threshold = 500
        
class FinalSharedLs9(FinalSharedLs): # not very good
    def __init__(self):
        super().__init__()
        self.filter_outliers = False
        self.gen_loss = 'l2'
        self.clip_gradient_threshold = 10


class VaeEpsilonSpace(MlpShapeFeaturesSim):
    def __init__(self):
        super().__init__()
        self.use_epsilon_space = True
        self.gen_loss_weight = 2000
        self.use_outpos_statistics = False

class VaeEpsilonSpace2(VaeEpsilonSpace):
    def __init__(self):
        super().__init__()
        self.gen_loss_weight = 5000

class VaeEpsilonSpace3(VaeEpsilonSpace):
    def __init__(self):
        super().__init__()
        self.gen_loss_weight = 10000

class VaeEpsilonSpace4(VaeEpsilonSpace):
    def __init__(self):
        super().__init__()
        self.gen_loss_weight = 500

class VaeAzimuth(VaeEpsilonSpace3):
    def __init__(self):
        super().__init__()
        self.shape_features_name = 'mlsPolyAS3'
        self.polynomial_space = 'AS'
        self.prediction_space = 'AS'

class VaeEpsilonSpace5(VaeEpsilonSpace): # Bad, not enough gen_loss_weight
    def __init__(self):
        super().__init__()
        self.gen_loss_weight = 100

class VaeEpsilonSpace6(VaeEpsilonSpace): # Bad, not enough gen_loss_weight
    def __init__(self):
        super().__init__()
        self.gen_loss_weight = 10



class VaeSharedSimNoOutStats(VaeFeaturePreSharedSim2):
    def __init__(self):
        super().__init__()
        self.use_outpos_statistics = False


class VaeSharedSimFlt1(VaeFeaturePreSharedSim2):
    def __init__(self):
        super().__init__()
        self.outlier_distance = 10.0


class VaeSharedSimFlt2(VaeFeaturePreSharedSim2):
    def __init__(self):
        super().__init__()
        self.outlier_distance = 3.0


class VaeSharedSimSkip(VaeFeaturePreSharedSim2):
    def __init__(self):
        super().__init__()
        self.first_layer_feats = False

class VaeFeaturePreSharedSim2L2(VaeFeaturePreSharedSim2):
    def __init__(self):
        super().__init__()
        self.n_latent = 2

class VaeFeaturePreSharedSim2L3(VaeFeaturePreSharedSim2):
    def __init__(self):
        super().__init__()
        self.n_latent = 3

class VaeFeaturePreSharedSim2L8(VaeFeaturePreSharedSim2):
    def __init__(self):
        super().__init__()
        self.n_latent = 8

class VaeFeaturePreSharedSim2L16(VaeFeaturePreSharedSim2):
    def __init__(self):
        super().__init__()
        self.n_latent = 16


class VaeFeaturePreSharedSim2G2000(VaeFeaturePreSharedSim2):
    def __init__(self):
        super().__init__()
        self.gen_loss_weight = 2000

class VaeFeaturePreSharedSim2G500(VaeFeaturePreSharedSim2):
    def __init__(self):
        super().__init__()
        self.gen_loss_weight = 500


class VaeFeaturePreSharedSim2G250(VaeFeaturePreSharedSim2):
    def __init__(self):
        super().__init__()
        self.gen_loss_weight = 250

class VaeFeaturePreSharedSim2L2(VaeFeaturePreSharedSim2):
    def __init__(self):
        super().__init__()
        self.gen_loss = 'l2'

class VaeFeaturePreSharedSim2L1(VaeFeaturePreSharedSim2):
    def __init__(self):
        super().__init__()
        self.gen_loss = 'l1'
        self.gen_loss_weight = 2000

class VaeFeaturePreSharedSim2L1G4000(VaeFeaturePreSharedSim2):
    def __init__(self):
        super().__init__()
        self.gen_loss = 'l1'
        self.gen_loss_weight = 4000

class VaeFeaturePreSharedSim2NoOutlierReject(VaeFeaturePreSharedSim2):
    def __init__(self):
        super().__init__()
        self.filter_outliers = False
        self.clip_gradient_threshold = 10

class VaeFeaturePreSharedSim2ScaledLoss(VaeFeaturePreSharedSim2):
    def __init__(self):
        super().__init__()
        self.scale_loss_by_kernel_epsilon = True


class VaeFeaturePreSharedSim2ScaledLoss2(VaeFeaturePreSharedSim2):
    def __init__(self):
        super().__init__()
        self.scale_loss_by_kernel_epsilon = True
        self.gen_loss_weight = 4000

class VaeFeaturePreSharedSim2ScaledLoss3(VaeFeaturePreSharedSim2):
    def __init__(self):
        super().__init__()
        self.scale_loss_by_kernel_epsilon = True
        self.gen_loss_weight = 100

class MlpShapeFeaturesHardConstraint(MlpShapeFeatures):
    def __init__(self):
        super().__init__()
        self.dataset = SCATTERDATASURFACECONSTRAINT


class VaeEfficientHardConstraint(VaeEfficient):
    def __init__(self):
        super().__init__()
        self.dataset = SCATTERDATASURFACECONSTRAINT


class VaeEfficientNew1(VaeEfficient):
    def __init__(self):
        super().__init__()
        self.n_mlp_layers = 5


class VaeEfficientResnetPre(VaeEfficient):
    def __init__(self):
        super().__init__()
        self.use_batch_norm = True
        self.shape_feat_net = lambda x, is_training: models.nn.multilayer_resnet(
            x, is_training, None, 2, 64, self.use_batch_norm, 'shapemlp')


class VaeEfficientResnetPre2(VaeEfficient):
    def __init__(self):
        super().__init__()
        self.shape_feat_net = lambda x, is_training: models.nn.multilayer_resnet(
            x, is_training, None, 3, 32, self.use_batch_norm, 'shapemlp')


class VaeEfficientNew2(VaeEfficient):
    def __init__(self):
        super().__init__()
        self.shape_feat_net = lambda x, is_training: models.nn.multilayer_fcn(
            x, is_training, None, 3, 64, self.use_batch_norm, 'shapemlp')


class VaeEfficientNew3(VaeEfficient):
    def __init__(self):
        super().__init__()
        self.shape_feat_net = lambda x, is_training: models.nn.multilayer_fcn(
            x, is_training, None, 4, 64, self.use_batch_norm, 'shapemlp')


class VaeEfficientNew4(VaeEfficient):
    def __init__(self):
        super().__init__()
        self.shape_feat_net = lambda x, is_training: models.nn.multilayer_fcn(
            x, is_training, None, 5, 64, self.use_batch_norm, 'shapemlp')


class VaeEfficientNew5(VaeEfficient):
    def __init__(self):
        super().__init__()
        self.shape_feat_net = lambda x, is_training: models.nn.multilayer_fcn(
            x, is_training, None, 5, 32, self.use_batch_norm, 'shapemlp')


class VaeEfficientNew6(VaeEfficient):
    def __init__(self):
        super().__init__()
        self.shape_feat_net = lambda x, is_training: models.nn.multilayer_fcn(
            x, is_training, None, 2, 64, self.use_batch_norm, 'shapemlp')


class VaeEfficientL500(VaeEfficient):
    def __init__(self):
        super().__init__()
        self.gen_loss_weight = 500


class VaeEfficientL100(VaeEfficient):
    def __init__(self):
        super().__init__()
        self.gen_loss_weight = 100


class VaeEfficientL2000(VaeEfficient):
    def __init__(self):
        super().__init__()
        self.gen_loss_weight = 2000


class VaeFunnel3(MlpShapeFeatures):
    def __init__(self):
        super().__init__()
        self.n_latent = 4
        self.n_mlp_width = [64, 32, 16, 8, 4]
        self.n_mlp_layers = 5
        self.first_layer_feats = True


class VaeFunnel4(MlpShapeFeatures):
    def __init__(self):
        super().__init__()
        self.n_latent = 4
        self.n_mlp_width = [64, 32, 16, 8]
        self.n_mlp_layers = 4
        self.first_layer_feats = True


class VaeShCoefficients(MlpShapeFeatures):
    def __init__(self):
        super().__init__()
        self.dataset = SCATTERDATASH
        self.use_sh_coeffs = True
        self.shape_features_name = 'shCoeffs'
        self.pass_in_dir = True

# class VaeShCoefficientsRotated(VaeShCoefficients):
#     def __init__(self):
#         super().__init__()
#         self.rotate_features = True


class VaeShCoeffsShared(VaeShCoefficients):
    def __init__(self):
        super().__init__()
        self.shape_feat_net = vae.model.shared_preproc_mlp


# class VaeShCoeffsSharedRotated(VaeShCoefficientsRotated):
#     def __init__(self):
#         super().__init__()
#         self.shape_feat_net = vae.model.shared_preproc_mlp


class VaeLegendre(MlpShapeFeatures):
    def __init__(self):
        super().__init__()
        self.shape_features_name = 'legendrePolyLS3'


class VaeLegendreDeg2(MlpShapeFeatures):
    def __init__(self):
        super().__init__()
        self.shape_features_name = 'legendrePolyLS2'
        self.dataset = SCATTERDATADEG2


class VaeNoKdThreshold(MlpShapeFeatures):
    def __init__(self):
        super().__init__()
        self.dataset = '0063_ScatterDataNoThreshold'


class VaeNoKdThresholdDeg2(MlpShapeFeaturesDeg2):
    def __init__(self):
        super().__init__()
        self.dataset = '0065_ScatterDataNoThresholdDeg2'


class VaeMoreDataReg(MlpShapeFeatures):
    def __init__(self):
        super().__init__()
        self.dataset = '0064_ScatterDataMoreReg'


class VaeEvalPoly(MlpShapeFeatures):
    def __init__(self):
        super().__init__()
        self.eval_poly_feature = True


class VaeEvalPolySigmoid(MlpShapeFeatures):
    def __init__(self):
        super().__init__()
        self.eval_poly_feature = True
        self.sigmoid_features = True


class VaeEvalPolyBinary(MlpShapeFeatures):
    def __init__(self):
        super().__init__()
        self.binary_features = True
        self.eval_poly_feature = True


class VaeEvalPolyCnn(MlpShapeFeatures):
    def __init__(self):
        super().__init__()
        self.binary_features = False
        self.eval_poly_feature = True

        self.poly_eval_range = 2.0
        self.poly_eval_steps = 8
        self.poly_conv_net = vae.model.polyCnn


class VaeEvalPolyBinaryCnn(VaeEvalPolyCnn):
    def __init__(self):
        super().__init__()
        self.binary_features = True


class VaeEvalPolySigmoidCnn(VaeEvalPolyCnn):
    def __init__(self):
        super().__init__()
        self.sigmoid_features = True


class VaeRescaled(MlpShapeFeatures):
    def __init__(self):
        super().__init__()
        self.scale_point_by_poly_scale = True


class VaeEfficient2(MlpShapeFeatures):
    def __init__(self):
        super().__init__()
        self.n_latent = 4
        self.first_layer_feats = True
        self.n_mlp_layers = 5
        self.n_mlp_width = 32


class VaeEfficient3(MlpShapeFeatures):
    def __init__(self):
        super().__init__()
        self.n_latent = 4
        self.first_layer_feats = True
        self.n_mlp_layers = 3
        self.n_mlp_width = 32
        self.shape_feat_net = None


class VaeEfficient4(MlpShapeFeatures):
    def __init__(self):
        super().__init__()
        self.n_latent = 4
        self.first_layer_feats = True
        self.n_mlp_layers = 5
        self.n_mlp_width = 16
        self.shape_feat_net = None


class VaeEfficientNoPreproc(VaeEfficient):
    def __init__(self):
        super().__init__()
        self.shape_feat_net = None


class VaeDummy(MlpShapeFeatures):
    def __init__(self):
        super().__init__()
        self.n_latent = 4
        self.first_layer_feats = True
        self.shape_feat_net = None
        self.n_mlp_layers = 0
        self.n_mlp_width = 8


class VaeRotatedFeatures(MlpShapeFeatures):
    def __init__(self):
        super().__init__()
        self.shape_features_name = 'mlsPolyTS3'
        self.polynomial_space = 'TS'
        self.rotate_features = True
        self.pass_in_dir = False

        self.shape_feat_net = lambda x, is_training: models.nn.multilayer_fcn(
            x, is_training, None, 3, 32, self.use_batch_norm, 'shapemlp', n_out=36)


class VaeRotatedFeatures2(VaeRotatedFeatures):
    def __init__(self):
        super().__init__()
        self.n_latent = 4
        self.first_layer_feats = True
        self.n_mlp_layers = 2
        self.n_mlp_width = 32
        self.shape_feat_net = lambda x, is_training: models.nn.multilayer_fcn(
            x, is_training, None, 3, 64, self.use_batch_norm, 'shapemlp', n_out=48)


class VaeRotatedFeatures3(VaeRotatedFeatures):
    def __init__(self):
        super().__init__()
        self.n_latent = 4
        self.first_layer_feats = True
        self.n_mlp_layers = 1
        self.n_mlp_width = 32
        self.shape_feat_net = lambda x, is_training: models.nn.multilayer_fcn(
            x, is_training, None, 3, 64, self.use_batch_norm, 'shapemlp', n_out=48)


class VaeRotatedFeaturesPassInDir(VaeRotatedFeatures):
    def __init__(self):
        super().__init__()
        self.shape_features_name = 'mlsPolyTS3'
        self.polynomial_space = 'TS'
        self.rotate_features = True
        self.pass_in_dir = True


class PointNetBaseline(MlpShapeFeatures):
    def __init__(self):
        super().__init__()
        self.use_point_net = True
        self.n_point_net_points = 64
        self.dataset = '0061_ScatterDataSpherePc'
        self.shape_features_name = None


class PointNetFullData(MlpShapeFeatures):
    def __init__(self):
        super().__init__()
        self.use_point_net = True
        self.n_point_net_points = 64
        self.dataset = '0062_ScatterDataPc'
        self.shape_features_name = None


class PointNetNormals(PointNetFullData):
    def __init__(self):
        super().__init__()
        self.point_net_use_normals = True


class PointNetSphereNormals(PointNetBaseline):
    def __init__(self):
        super().__init__()
        self.point_net_use_normals = True


class PointNetWeighted(PointNetFullData):
    def __init__(self):
        super().__init__()
        self.point_net_use_normals = True
        self.point_net_use_weights = True


class PointNetSphereWeighted(PointNetSphereNormals):
    def __init__(self):
        super().__init__()
        self.point_net_use_weights = True


class PointNetTrue(MlpShapeFeatures):
    def __init__(self):
        super().__init__()
        self.use_point_net = True
        self.n_point_net_points = 64
        self.dataset = SCATTERDATATRUE
        self.shape_features_name = None
        self.point_net_use_normals = True
        self.point_net_use_weights = False


class PointNetRescaled(PointNetTrue):
    def __init__(self):
        super().__init__()
        self.scale_point_by_poly_scale = True


class PointNet2(PointNetTrue):
    def __init__(self):
        super().__init__()
        self.point_net_feature_sizes = [16, 32, 64]


class PointNet3(PointNetTrue):
    def __init__(self):
        super().__init__()
        self.point_net_feature_sizes = [32, 32, 32]


class PointNet4(PointNetTrue):
    def __init__(self):
        super().__init__()
        self.point_net_feature_sizes = [16, 32, 64, 64]


class PointNetShared(PointNetTrue):
    def __init__(self):
        super().__init__()
        self.point_net = vae.model.baseline_point_net_shared


class VaePointnetHist(PointNetTrue):
    def __init__(self):
        super().__init__()
        self.point_net_normal_histogram = True
        self.point_net_n_normal_bins = 3
        self.point_net_feature_sizes = [8, 16, 16]
        self.point_net_use_weights = False


class VaePointnetHist5(VaePointnetHist):
    def __init__(self):
        super().__init__()
        self.point_net_n_normal_bins = 5


class ProjectiveBaseline(MlpShapeFeatures):
    def __init__(self):
        super().__init__()
        self.model = vae.model.projectiveNet
        self.n_coupling_layers = 5
        self.n_latent = 3


class VaeFunnel(MlpShapeFeatures):
    def __init__(self):
        super().__init__()
        self.n_latent = 4
        self.n_mlp_width = [64, 32, 16, 8, 4]
        self.n_mlp_layers = 5
        self.first_layer_feats = True
        self.shape_feat_net = None


class VaeFunnel2(MlpShapeFeatures):
    def __init__(self):
        super().__init__()
        self.n_latent = 4
        self.n_mlp_width = [32, 16, 8, 4]
        self.n_mlp_layers = 4
        self.first_layer_feats = True
        self.shape_feat_net = None


class VaeTsFeatures(MlpShapeFeatures):
    def __init__(self):
        super().__init__()
        self.shape_features_name = 'mlsPolyTS3'
        self.polynomial_space = 'TS'
        self.pass_in_dir = True


class VaeTsFeaturesAngleAfterPreproc(MlpShapeFeatures):
    def __init__(self):
        super().__init__()
        self.shape_features_name = 'mlsPolyTS3'
        self.polynomial_space = 'TS'
        self.pass_in_dir_after_preprocess = True
        self.pass_in_dir = False


class VaeFirstLayerFeats(MlpShapeFeatures):
    def __init__(self):
        super().__init__()
        self.first_layer_feats = True


class VaeFirstLayerFeatsPre(MlpShapeFeatures):
    def __init__(self):
        super().__init__()
        self.first_layer_feats = True
        self.shape_feat_net = vae.model.shared_preproc_mlp


class MlpFilter(MlpShapeFeatures):
    def __init__(self):
        super().__init__()
        self.filter_feature_outliers = True


class MlpNoOutlierReject(MlpShapeFeatures):
    def __init__(self):
        super().__init__()
        self.filter_outliers = False
        self.filter_feature_outliers = False


class MlpL2NoOutlier(MlpShapeFeatures):  # This should be the least robust, but we still clip gradients, so its okay?
    def __init__(self):
        super().__init__()
        self.filter_outliers = False
        self.filter_feature_outliers = False
        self.gen_loss = 'l2'


class MlpL2(MlpShapeFeatures):  # This should be the least robust, but we still clip gradients, so its okay?
    def __init__(self):
        super().__init__()
        self.gen_loss = 'l2'


class BaselineNICE(MlpShapeFeatures):
    def __init__(self):
        super().__init__()
        self.model = vae.model.baselineNICE
        self.use_nice = True
        self.n_latent = 3
        self.batch_size = 128
        self.learningrate = 2e-4

        self.n_coupling_layers = 12

        self.clip_gradient_threshold = 100
        self.filter_outliers = True

        self.use_res_net = False
        self.use_legacy_res_net = True


class NiceNewResnet(BaselineNICE):
    def __init__(self):
        super().__init__()
        self.use_res_net = True
        self.use_legacy_res_net = False


class NiceNoOutlierReject(BaselineNICE):
    def __init__(self):
        super().__init__()
        self.filter_outliers = False
        self.filter_feature_outliers = False


class NiceRegularized(BaselineNICE):
    def __init__(self):
        super().__init__()
        self.regularizer = tf.contrib.layers.l2_regularizer(0.00001)


class NiceRegularized2(BaselineNICE):
    def __init__(self):
        super().__init__()
        self.regularizer = tf.contrib.layers.l2_regularizer(0.001)


class NiceCoordOffset(BaselineNICE):
    def __init__(self):
        super().__init__()
        self.use_coupling_coord_offset = True


class NiceClampShapeFeat(BaselineNICE):
    def __init__(self):
        super().__init__()
        self.clamp_shape_features = True


class MlpClampShapeFeat(MlpShapeFeatures):
    def __init__(self):
        super().__init__()
        self.clamp_shape_features = True


class BaselineNICEClipMore(BaselineNICE):
    def __init__(self):
        super().__init__()
        self.clip_gradient_threshold = 10


class NiceFirstLayerFeats(BaselineNICE):
    def __init__(self):
        super().__init__()
        self.nice_first_layer_feats = True


class NiceFeaturePre(BaselineNICE):  # Works best (makes the network more complex overall)
    def __init__(self):
        super().__init__()
        self.shape_feat_net = lambda x, is_training: models.nn.multilayer_fcn(
            x, is_training, None, 3, 32, self.use_batch_norm, 'shapemlp')
        self.nice_first_layer_feats = True


class NiceFeaturePreShared(BaselineNICE):  # Works best (makes the network more complex overall)
    def __init__(self):
        super().__init__()
        self.shape_feat_net = vae.model.shared_preproc_mlp
        self.nice_first_layer_feats = True


class NiceFeaturePreSimple(NiceFeaturePre):  # Not as good as the other preproc, too simple
    def __init__(self):
        super().__init__()
        self.shape_feat_net = lambda x, is_training: models.nn.multilayer_fcn(
            x, is_training, None, 3, 32, self.use_batch_norm, 'shapemlp')
        self.n_mlp_layers = 3
        self.n_mlp_width = 32


class NiceFeaturePreRes(NiceFeaturePre):  # Works well
    def __init__(self):
        super().__init__()
        self.shape_feat_net = lambda x, is_training: models.nn.multilayer_resnet(
            x, is_training, None, 3, 64, self.use_batch_norm, 'shapemlp')
        self.n_mlp_layers = 3
        self.n_mlp_width = 32


class NiceFeaturePreRes2(NiceFeaturePre):  # Not as good as the other preproc, too simple
    def __init__(self):
        super().__init__()
        self.shape_feat_net = lambda x, is_training: models.nn.multilayer_resnet(
            x, is_training, None, 2, 32, self.use_batch_norm, 'shapemlp')
        self.n_mlp_layers = 3
        self.n_mlp_width = 32


class NiceFeaturePreResSimple(NiceFeaturePre):  # Works reasonably well
    def __init__(self):
        super().__init__()
        self.shape_feat_net = lambda x, is_training: models.nn.multilayer_resnet(
            x, is_training, None, 3, 64, self.use_batch_norm, 'shapemlp')
        self.n_mlp_layers = 2
        self.n_mlp_width = 32


class VaeFeaturePre(MlpShapeFeatures):  # Works best (makes the network more complex overall)
    def __init__(self):
        super().__init__()
        self.shape_feat_net = lambda x, is_training: models.nn.multilayer_fcn(
            x, is_training, None, 3, 32, self.use_batch_norm, 'shapemlp')


class VaeFeaturePreShared(MlpShapeFeatures):  # Works best (makes the network more complex overall)
    def __init__(self):
        super().__init__()
        self.shape_feat_net = vae.model.shared_preproc_mlp

class VaeFeaturePreShared2(MlpShapeFeatures):  # Works best (makes the network more complex overall)
    def __init__(self):
        super().__init__()
        self.shape_feat_net = vae.model.shared_preproc_mlp_2

class VaeFeaturePreSharedDeg2(MlpShapeFeaturesDeg2):  # Works best (makes the network more complex overall)
    def __init__(self):
        super().__init__()
        self.shape_feat_net = vae.model.shared_preproc_mlp


class VaeFeaturePreSimple(VaeFeaturePre):  # Not as good as the other preproc, too simple
    def __init__(self):
        super().__init__()
        self.shape_feat_net = lambda x, is_training: models.nn.multilayer_fcn(
            x, is_training, None, 3, 32, self.use_batch_norm, 'shapemlp')
        self.n_mlp_layers = 3
        self.n_mlp_width = 32


class VaeFeaturePreRes(VaeFeaturePre):  # Works well
    def __init__(self):
        super().__init__()
        self.shape_feat_net = lambda x, is_training: models.nn.multilayer_resnet(
            x, is_training, None, 3, 64, self.use_batch_norm, 'shapemlp')
        self.n_mlp_layers = 3
        self.n_mlp_width = 32


class VaeFeaturePreRes2(VaeFeaturePre):  # Not as good as the other preproc, too simple
    def __init__(self):
        super().__init__()
        self.shape_feat_net = lambda x, is_training: models.nn.multilayer_resnet(
            x, is_training, None, 2, 32, self.use_batch_norm, 'shapemlp')
        self.n_mlp_layers = 3
        self.n_mlp_width = 32


class VaeFeaturePreResSimple(VaeFeaturePre):  # Works reasonably well
    def __init__(self):
        super().__init__()
        self.shape_feat_net = lambda x, is_training: models.nn.multilayer_resnet(
            x, is_training, None, 3, 64, self.use_batch_norm, 'shapemlp')
        self.n_mlp_layers = 2
        self.n_mlp_width = 32


class NiceSigmoidFeatures(BaselineNICE):  # Seems worse than regular features
    def __init__(self):
        super().__init__()
        self.sigmoid_features = True


class NiceMlp(BaselineNICE):
    def __init__(self):
        super().__init__()
        self.use_res_net = False
        self.use_legacy_res_net = False


class NiceC15(BaselineNICE):
    def __init__(self):
        super().__init__()
        self.n_coupling_layers = 15


class NiceC21(BaselineNICE):
    def __init__(self):
        super().__init__()
        self.n_coupling_layers = 21


class NiceC6(BaselineNICE):
    def __init__(self):
        super().__init__()
        self.n_coupling_layers = 6


class NiceDeep(BaselineNICE):
    def __init__(self):
        super().__init__()
        self.n_mlp_layers = 5


class NiceSimple(BaselineNICE):
    def __init__(self):
        super().__init__()
        self.n_mlp_layers = 3
        self.n_mlp_width = 32


class NiceSimpler(BaselineNICE):
    def __init__(self):
        super().__init__()
        self.n_mlp_layers = 2
        self.n_mlp_width = 32


class NiceFixedMedium(BaselineNICE):
    def __init__(self):
        super().__init__()
        self.dataset = vae.datapipeline.ScatterDataFixedMedium


class NiceFixedMediumSearchlight(BaselineNICE):
    def __init__(self):
        super().__init__()
        self.dataset = vae.datapipeline.ScatterDataFixedMediumSearchlight


class NiceDeg2(BaselineNICE):
    def __init__(self):
        super().__init__()
        self.shape_features_name = 'mlsPolyLS2'
        self.dataset = SCATTERDATADEG2


class NiceDeg4(BaselineNICE):
    def __init__(self):
        super().__init__()
        self.shape_features_name = 'mlsPolyLS4'


class NiceIsData(BaselineNICE):
    def __init__(self):
        super().__init__()
        self.dataset = SCATTERDATAIS


class NiceSphereDeg3(BaselineNICE):
    def __init__(self):
        super().__init__()
        self.shape_features_name = 'mlsPolyLS3'
        self.dataset = vae.datapipeline.ScatterDataSphere
        self.num_epochs = 100


class NiceSphereFixedMediumDeg2(BaselineNICE):
    def __init__(self):
        super().__init__()
        self.shape_features_name = 'mlsPolyLS2'
        self.dataset = vae.datapipeline.ScatterDataSphereFixedMedium
        self.num_epochs = 100


class NiceSphereFixedMediumDeg3(BaselineNICE):
    def __init__(self):
        super().__init__()
        self.dataset = SCATTERDATASPHEREFIXEDMEDIUM
        self.num_epochs = 100


class NiceSphereFixedMediumSearchlightDeg2(BaselineNICE):
    def __init__(self):
        super().__init__()
        self.shape_features_name = 'mlsPolyLS2'
        self.dataset = SCATTERDATASPHEREFIXEDMEDIUMSEARCHLIGHTDEG2
        self.num_epochs = 100


class NiceSphereFixedMediumSearchlightDeg3(BaselineNICE):
    def __init__(self):
        super().__init__()
        self.shape_features_name = 'mlsPolyLS3'
        self.dataset = vae.datapipeline.ScatterDataSphereFixedMediumSearchlight
        self.num_epochs = 100


class NiceLrLarge(BaselineNICE):

    def __init__(self):
        super().__init__()
        self.learningrate = 2e-4


class NiceLrLargeConst(BaselineNICE):

    def __init__(self):
        super().__init__()
        self.learningrate = 2e-4
        self.use_adaptive_lr = False


class MlpShapeFeaturesConstLR(MlpShapeFeatures):
    def __init__(self):
        super().__init__()
        self.use_adaptive_lr = False


class MlpShapeFeaturesSigmoid(MlpShapeFeatures):
    def __init__(self):
        super().__init__()
        self.sigmoid_features = True

        self.learningrate = self.learningrate * 0.1
        self.loss_clamp_val = 100


class MlpShapeFeaturesSigmoidClampMore(MlpShapeFeaturesSigmoid):
    def __init__(self):
        super().__init__()
        self.loss_clamp_val = 20


class MlpShapeFeaturesBatch128(MlpShapeFeatures):
    def __init__(self):
        super().__init__()
        self.batch_size = 128


class MlpShapeFeaturesBatch64(MlpShapeFeatures):
    def __init__(self):
        super().__init__()
        self.batch_size = 64


class MlpShapeFeaturesBatch64Clamp(MlpShapeFeaturesBatch64):
    def __init__(self):
        super().__init__()
        self.loss_clamp_val = 100


class MlpShapeFeaturesBatch64ClampMore(MlpShapeFeaturesBatch64):
    def __init__(self):
        super().__init__()
        self.loss_clamp_val = 20


class MlpShapeFeaturesBatch16(MlpShapeFeatures):
    def __init__(self):
        super().__init__()
        self.batch_size = 16


class MlpShapeFeaturesBatch8(MlpShapeFeatures):
    def __init__(self):
        super().__init__()
        self.batch_size = 8


class SphereDeg2(MlpShapeFeatures):
    def __init__(self):
        super().__init__()
        self.shape_features_name = 'mlsPolyLS2'
        self.dataset = SCATTERDATASPHEREDEG2
        self.num_epochs = 100


class NiceSphereDeg2(BaselineNICE):
    def __init__(self):
        super().__init__()
        self.shape_features_name = 'mlsPolyLS2'
        self.dataset = SCATTERDATASPHEREDEG2
        self.num_epochs = 100


class SphereBigFixedMedium(MlpShapeFeatures):
    def __init__(self):
        super().__init__()
        self.dataset = SCATTERDATABIGSPHEREFIXEDMEDIUM
        self.num_epochs = 100


class NiceSphereBigFixedMedium(BaselineNICE):
    def __init__(self):
        super().__init__()
        self.dataset = SCATTERDATABIGSPHEREFIXEDMEDIUM
        self.num_epochs = 100


class SphereDeg2Deep(SphereDeg2):
    def __init__(self):
        super().__init__()
        self.n_mlp_layers = 4


class SphereDeg2Wide(SphereDeg2):
    def __init__(self):
        super().__init__()
        self.n_mlp_width = 128


class SphereDeg2L100(SphereDeg2):
    def __init__(self):
        super().__init__()
        self.gen_loss_weight = 100


class SphereDeg2L10(SphereDeg2):
    def __init__(self):
        super().__init__()
        self.gen_loss_weight = 10


class SphereDeg2L500(SphereDeg2):
    def __init__(self):
        super().__init__()
        self.gen_loss_weight = 500


class SphereDeg2L2000(SphereDeg2):
    def __init__(self):
        super().__init__()
        self.gen_loss_weight = 2000


class SphereDeg2L5000(SphereDeg2):
    def __init__(self):
        super().__init__()
        self.gen_loss_weight = 5000


class SphereDeg2L200(SphereDeg2):
    def __init__(self):
        super().__init__()
        self.gen_loss_weight = 500


class SphereDeg3(MlpShapeFeatures):
    def __init__(self):
        super().__init__()
        self.dataset = vae.datapipeline.ScatterDataSphere
        self.num_epochs = 100


class SphereFixedMediumDeg2(MlpShapeFeatures):
    def __init__(self):
        super().__init__()
        self.shape_features_name = 'mlsPolyLS2'
        self.dataset = vae.datapipeline.ScatterDataSphereFixedMedium
        self.num_epochs = 100


class SphereFixedMediumDeg3(MlpShapeFeatures):
    def __init__(self):
        super().__init__()
        self.dataset = SCATTERDATASPHEREFIXEDMEDIUM
        self.num_epochs = 100


class SphereFixedMediumSearchlightDeg2(MlpShapeFeatures):
    def __init__(self):
        super().__init__()
        self.shape_features_name = 'mlsPolyLS2'
        self.dataset = SCATTERDATASPHEREFIXEDMEDIUMSEARCHLIGHTDEG2
        self.num_epochs = 100


class MlpProjectGd(MlpShapeFeatures):
    def __init__(self):
        super().__init__()
        self.surface_projection_method = 'gd'
        self.surface_projection_iterations = 5


class MlpProjectGdSphereFixedSearchLightDeg2(SphereFixedMediumSearchlightDeg2):
    def __init__(self):
        super().__init__()
        self.surface_projection_method = 'gd'
        self.surface_projection_iterations = 5


class MlpProjectGdIter1(MlpProjectGd):
    def __init__(self):
        super().__init__()
        self.surface_projection_iterations = 1


class MlpProjectGdSphereFixedSearchLightDeg2Iter1(MlpProjectGdSphereFixedSearchLightDeg2):
    def __init__(self):
        super().__init__()
        self.surface_projection_iterations = 1


class MlpProjectGdIter3(MlpProjectGd):
    def __init__(self):
        super().__init__()
        self.surface_projection_iterations = 3


class MlpProjectGdSphereFixedSearchLightDeg2Iter3(MlpProjectGdSphereFixedSearchLightDeg2):
    def __init__(self):
        super().__init__()
        self.surface_projection_iterations = 3


class SphereFixedMediumSearchlightDeg3(MlpShapeFeatures):
    def __init__(self):
        super().__init__()
        self.dataset = vae.datapipeline.ScatterDataSphereFixedMediumSearchlight
        self.num_epochs = 100


class PlaneFixedMediumSearchlight(MlpShapeFeatures):
    def __init__(self):
        super().__init__()
        self.shape_features_name = 'mlsPolyLS1'
        self.dataset = SCATTERDATAPLANEFIXEDMEDIUMSEARCHLIGHT
        self.num_epochs = 100


class MlpOffSurfacePenalty(MlpShapeFeatures):
    def __init__(self):
        super().__init__()
        self.off_surface_penalty_weight = 0.1


class MlpOffSurfacePenalty2(MlpShapeFeatures):
    def __init__(self):
        super().__init__()
        self.off_surface_penalty_weight = 0.01


class MlpOffSurfacePenalty3(MlpShapeFeatures):
    def __init__(self):
        super().__init__()
        self.off_surface_penalty_weight = 1.0


class MlpOffSurfacePenalty4(MlpShapeFeatures):
    def __init__(self):
        super().__init__()
        self.off_surface_penalty_weight = 10.0


tmp = [
    MlpOffSurfacePenalty,
    MlpOffSurfacePenalty2,
    MlpOffSurfacePenalty3,
    MlpOffSurfacePenalty4,
]
vae.utils.add_variants(globals(), tmp, {'off_surface_penalty_clamp': 10.0}, 'Clamped')


class SphereOffSurfacePenalty(SphereFixedMediumSearchlightDeg2):
    def __init__(self):
        super().__init__()
        self.off_surface_penalty_weight = 0.1


class SphereOffSurfacePenalty2(SphereFixedMediumSearchlightDeg2):
    def __init__(self):
        super().__init__()
        self.off_surface_penalty_weight = 0.01


class SphereOffSurfacePenalty3(SphereFixedMediumSearchlightDeg2):
    def __init__(self):
        super().__init__()
        self.off_surface_penalty_weight = 1.0


class SphereOffSurfacePenalty4(SphereFixedMediumSearchlightDeg2):
    def __init__(self):
        super().__init__()
        self.off_surface_penalty_weight = 10.0


class NicePlaneFixedMediumSearchlight(BaselineNICE):
    def __init__(self):
        super().__init__()
        self.shape_features_name = 'mlsPolyLS1'
        self.dataset = SCATTERDATAPLANEFIXEDMEDIUMSEARCHLIGHT
        self.num_epochs = 100
        self.learningrate = 0.00001

        self.clip_gradient_threshold = 100
        self.filter_outliers = True
        self.use_res_net = False
        self.use_legacy_res_net = True


class NicePlaneNoOutlierRejection(NicePlaneFixedMediumSearchlight):
    def __init__(self):
        super().__init__()
        self.filter_outliers = False


class NicePlaneNewResnet(NicePlaneFixedMediumSearchlight):
    def __init__(self):
        super().__init__()
        self.use_res_net = True
        self.use_legacy_res_net = False


class NicePlaneMlp(NicePlaneFixedMediumSearchlight):
    def __init__(self):
        super().__init__()
        self.use_res_net = False
        self.use_legacy_res_net = False


class NicePlaneC15(NicePlaneFixedMediumSearchlight):
    def __init__(self):
        super().__init__()
        self.n_coupling_layers = 15


class NicePlaneC9(NicePlaneFixedMediumSearchlight):
    def __init__(self):
        super().__init__()
        self.n_coupling_layers = 9


class GaussianData(MlpShapeFeatures):
    def __init__(self):
        super().__init__()
        self.shape_features_name = 'mlsPoly3'
        self.dataset = vae.datapipeline.GaussianData
        self.num_epochs = 100
        self.n_latent = 3
        self.predict_in_tangent_space = False
        self.prediction_space = 'WS'
        self.model = vae.model.standardVAE


class GaussianDataZ24(MlpShapeFeatures):
    def __init__(self):
        super().__init__()
        self.shape_features_name = 'mlsPoly3'
        self.dataset = vae.datapipeline.GaussianData
        self.num_epochs = 100
        self.n_latent = 24
        self.model = vae.model.standardVAE


class FixedMediumDeg3(MlpShapeFeatures):
    def __init__(self):
        super().__init__()
        self.dataset = vae.datapipeline.ScatterDataFixedMedium


class FixedMediumDeg2(MlpShapeFeatures):
    def __init__(self):
        super().__init__()
        self.dataset = vae.datapipeline.ScatterDataFixedMedium
        self.shape_features_name = 'mlsPolyLS2'


class FixedMediumSearchlightDeg3(MlpShapeFeatures):
    def __init__(self):
        super().__init__()
        self.dataset = vae.datapipeline.ScatterDataFixedMediumSearchlight


class FixedMediumSearchlightDeg2(MlpShapeFeatures):
    def __init__(self):
        super().__init__()
        self.dataset = vae.datapipeline.ScatterDataFixedMediumSearchlight
        self.shape_features_name = 'mlsPolyLS2'


class SphereFixedMediumDeg2FeatNoise(SphereFixedMediumDeg2):
    def __init__(self):
        super().__init__()
        self.add_noise_to_poly_coeffs = True
        self.feat_noise_variance = 0.001


class SphereFixedMediumDeg2FeatNoiseLowNoise(SphereFixedMediumDeg2):
    def __init__(self):
        super().__init__()
        self.add_noise_to_poly_coeffs = True
        self.feat_noise_variance = 0.0001


class SphereDeg2FeatNoise(SphereDeg2):
    def __init__(self):
        super().__init__()
        self.add_noise_to_poly_coeffs = True
        self.feat_noise_variance = 0.001


class SphereDeg2FeatNoiseLowNoise(SphereDeg2):
    def __init__(self):
        super().__init__()
        self.add_noise_to_poly_coeffs = True
        self.feat_noise_variance = 0.0001


tmp = [
    MlpShapeFeatures,
    SphereDeg2,
    SphereDeg3,
    SphereFixedMediumDeg2,
    SphereFixedMediumDeg3,
    SphereFixedMediumSearchlightDeg2,
    SphereFixedMediumSearchlightDeg3,
    PlaneFixedMediumSearchlight,
    FixedMediumDeg2,
    FixedMediumDeg3,
    FixedMediumSearchlightDeg2,
    FixedMediumSearchlightDeg3,
    GaussianDataZ24,
    GaussianData,
    SphereDeg2Deep,
    SphereDeg2Wide,
    SphereDeg2L100,
    SphereDeg2L10,
    SphereDeg2L500,
    SphereDeg2L2000,
    SphereDeg2L5000,
    SphereDeg2L200,
]
vae.utils.add_variants(globals(), tmp, {'latent_loss_annealing': 'linear'}, 'LinearAnneal')
vae.utils.add_variants(globals(), tmp, {'latent_loss_annealing': 'square'}, 'SquareAnneal')


class MlpShapeFeaturesRawAlbedo(MlpShapeFeatures):
    def __init__(self):
        super().__init__()
        self.use_eff_albedo = False


class MlpShapeFeaturesDeg2(MlpShapeFeatures):
    def __init__(self):
        super().__init__()
        self.dataset = SCATTERDATADEG2
        self.shape_features_name = 'mlsPolyLS2'


class MlpShapeFeaturesDeg4(MlpShapeFeatures):
    def __init__(self):
        super().__init__()
        self.shape_features_name = 'mlsPolyLS4'


class MlpShapeFeaturesDeg5(MlpShapeFeatures):
    def __init__(self):
        super().__init__()
        self.shape_features_name = 'mlsPolyLS5'


class ShallowShapeFeatures(MlpShapeFeatures):
    def __init__(self):
        super().__init__()
        self.use_res_net = True
        self.use_batch_norm = True
        self.n_mlp_layers = 2


class ShapeFeaturesW128(MlpShapeFeatures):
    def __init__(self):
        super().__init__()
        self.n_mlp_width = 128


class ShapeFeaturesW32(MlpShapeFeatures):
    def __init__(self):
        super().__init__()
        self.n_mlp_width = 32


class ShapeFeaturesW16(MlpShapeFeatures):
    def __init__(self):
        super().__init__()
        self.n_mlp_width = 16


class ShapeFeaturesD4(MlpShapeFeatures):
    def __init__(self):
        super().__init__()
        self.n_mlp_layers = 4


class DeepResShapeFeatures(MlpShapeFeatures):
    def __init__(self):
        super().__init__()
        self.use_res_net = True
        self.use_batch_norm = True
        self.n_mlp_layers = 5


class DeeperResShapeFeatures(MlpShapeFeatures):
    def __init__(self):
        super().__init__()
        self.use_res_net = True
        self.use_batch_norm = True
        self.n_mlp_layers = 7


class DeepResShapeFeaturesNoBn(MlpShapeFeatures):
    def __init__(self):
        super().__init__()
        self.use_res_net = True
        self.use_batch_norm = False
        self.n_mlp_layers = 5


class DeeperResShapeFeaturesNoBn(MlpShapeFeatures):
    def __init__(self):
        super().__init__()
        self.use_res_net = True
        self.use_batch_norm = False
        self.n_mlp_layers = 7


class ResShapeFeatures(MlpShapeFeatures):
    def __init__(self):
        super().__init__()
        self.use_res_net = True
        self.use_batch_norm = True


class BnShapeFeatures(MlpShapeFeatures):
    def __init__(self):
        super().__init__()
        self.use_batch_norm = True


class MlpShapeFeaturesL100(MlpShapeFeatures):
    def __init__(self):
        super().__init__()
        self.gen_loss_weight = 100


class MlpShapeFeaturesL500(MlpShapeFeatures):
    def __init__(self):
        super().__init__()
        self.gen_loss_weight = 500


class MlpShapeFeaturesL750(MlpShapeFeatures):
    def __init__(self):
        super().__init__()
        self.gen_loss_weight = 750


class MlpShapeFeaturesL1250(MlpShapeFeatures):
    def __init__(self):
        super().__init__()
        self.gen_loss_weight = 1250


class MlpShapeFeaturesL1500(MlpShapeFeatures):
    def __init__(self):
        super().__init__()
        self.gen_loss_weight = 1500


class MlpShapeFeaturesL2000(MlpShapeFeatures):
    def __init__(self):
        super().__init__()
        self.gen_loss_weight = 2000


class MlpShapeFeaturesZ2(MlpShapeFeatures):
    def __init__(self):
        super().__init__()
        self.n_latent = 2


class MlpShapeFeaturesZ3(MlpShapeFeatures):
    def __init__(self):
        super().__init__()
        self.n_latent = 3


class MlpShapeFeaturesZ4(MlpShapeFeatures):
    def __init__(self):
        super().__init__()
        self.n_latent = 4


class MlpShapeFeaturesZ8(MlpShapeFeatures):
    def __init__(self):
        super().__init__()
        self.n_latent = 8


class MlpShapeFeaturesZ16(MlpShapeFeatures):
    def __init__(self):
        super().__init__()
        self.n_latent = 16


class MlpShapeFeaturesZ32(MlpShapeFeatures):
    def __init__(self):
        super().__init__()
        self.n_latent = 32


class MlpShapeFeaturesSeed(MlpShapeFeatures):
    def __init__(self):
        super().__init__()
        self.seed = 51


class MlpShapeFeaturesSeed2(MlpShapeFeatures):
    def __init__(self):
        super().__init__()
        self.seed = 17


class MlpShapeFeaturesLargeLr(MlpShapeFeatures):
    def __init__(self):
        super().__init__()
        self.learningrate = 0.001


class MlpShapeFeaturesSmallLr(MlpShapeFeatures):
    def __init__(self):
        super().__init__()
        self.learningrate = 0.0001


class MlpShapeFeaturesSmallerLr(MlpShapeFeatures):
    def __init__(self):
        super().__init__()
        self.learningrate = 0.00005


class MlpShapeFeaturesAbsorptionDeep(MlpShapeFeatures):
    def __init__(self):
        super().__init__()
        self.n_abs_layers = 5


class MlpShapeFeaturesAbsorptionWide(MlpShapeFeatures):
    def __init__(self):
        super().__init__()
        self.n_abs_width = 64
