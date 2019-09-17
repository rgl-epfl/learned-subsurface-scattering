import inspect
import sys

import vae.datapipeline
import vae.model

# SCATTERDATA = '0081_ScatterData'
SCATTERDATA = '0004_ScatterData'
SCATTERDATASURFACECONSTRAINT = '0083_ScatterDataHardConstraint'
# SCATTERDATASIMILARITY = '0003_ScatterDataSimilarityMixed'
# SCATTERDATASIMILARITY = '0113_ScatterDataSimilarityMixed'
SCATTERDATASIMILARITY = '0121_ScatterData'
SCATTERDATASIMILARITYOLD = '0003_ScatterDataSimilarityMixed'

SCATTERDATAIS = '0053_ScatterDataIs'
SCATTERDATADEG2 = '0067_ScatterDataHighAbsDeg2'
SCATTERDATATRUE = '0068_ScatterDataPcTrue'
SCATTERDATASH = '0069_ScatterDataPcShTrue'
SCATTERDATASPHEREFIXEDMEDIUM = '0059_ScatterDataSphereFixedMedium'
SCATTERDATASPHEREFIXEDMEDIUMSEARCHLIGHT = '0111_ScatterDataSphereFixedMediumSearchlight'
SCATTERDATABIGSPHEREFIXEDMEDIUM = '0058_ScatterDataBigSphereFixedMedium'
SCATTERDATASPHERE = '0055_ScatterDataSphere'
SCATTERDATASPHEREDEG2 = '0056_ScatterDataSphereDeg2'
SCATTERDATASPHEREFIXEDMEDIUMSEARCHLIGHTDEG2 = '0060_ScatterDataSphereFixedMediumSearchlightDeg2'
SCATTERDATAPLANEFIXEDMEDIUMSEARCHLIGHT = '0016_ScatterDataPlaneFixedMediumSearchlight'

SCATTERDATASLABS = '0115_ScatterDataSlabMesh'

SCATTERDATAMIXED2 = '0122_ScatterDataMixed2'
SCATTERDATAMIXED3 = '0123_ScatterDataMixed3'
SCATTERDATAMIXED4 = '0124_ScatterDataMixed4'

class ConfigBase:
    def __init__(self):
        self.num_epochs = 50
        self.seed = 123
        self.learningrate = 0.0002
        self.model = None
        self.use_batch_norm = False
        self.batch_size = 32

        self.n_mlp_layers = 4
        self.n_mlp_width = 64
        self.use_cnn_features = True
        self.patchsize = 32
        self.n_latent = 24

        self.gen_loss_weight = 100

        self.use_res_net = False

        self.use_wae_mmd = False
        self.wae_random_enc = False
        self.add_encoder_noise = True  # Adds noise to the input of the encoder to reduce overfitting

        # TODO: Should be decoupled from dataset
        self.shape_features_name = None
        self.predict_in_tangent_space = False
        self.prediction_space = 'WS'
        self.polynomial_space = 'WS'
        # network to preprocess shape features before feeding to main network
        self.shape_feat_net = None

        self.optimizer = 'adam'
        self.use_adaptive_lr = True

        self.gen_loss = 'l2'
        self.loss_clamp_val = -1

        self.dim = 3

        self.abs_predict_diff = False
        self.abs_loss = 'l2'
        self.n_abs_layers = 3
        self.n_abs_width = 32
        self.absorption_model = vae.model.absorptionMlp
        self.abs_bounce_distribution = None
        self.abs_use_albedo_feature = True
        self.abs_regularizer = None
        self.abs_sigmoid_features = False
        self.abs_clamp_outlier_features = False
        self.abs_use_res_net = False
        self.use_eff_albedo = True

        self.dataset = SCATTERDATA

        self.latent_loss_annealing = None

        self.dropout_keep_prob = 1.0

        # Apply a sigmoid before passing features to NN
        self.sigmoid_features = False
        self.sigmoid_scale_factor = 1.0

        self.use_nice = False

        self.add_noise_to_poly_coeffs = False
        self.feat_noise_variance = 0.01
        self.ignore_features = False

        self.filter_outliers = False
        self.filter_feature_outliers = False

        self.clip_gradient_threshold = 1e6
        self.nice_first_layer_feats = False
        self.clamp_shape_features = False

        self.off_surface_penalty_weight = 0.0
        self.off_surface_penalty_clamp = -1

        self.surface_projection_method = None
        self.surface_projection_iterations = 5
        self.surface_projection_clip_range = 0.5

        self.pass_in_dir = False
        self.pass_in_dir_after_preprocess = False
        self.use_point_net = False
        self.n_point_net_points = 64
        self.point_net_feature_sizes = [8, 16, 32]
        self.point_net_use_normals = False
        self.point_net_use_weights = False
        # Normal binned point-net
        self.point_net_normal_histogram = False
        self.point_net_n_normal_bins = 3
        self.point_net = vae.model.baseline_point_net

        self.rotate_features = False

        self.scale_point_by_poly_scale = False

        self.binary_features = False
        self.eval_poly_feature = False

        self.poly_eval_range = 2.0
        self.poly_eval_steps = 3
        self.poly_conv_net = None

        self.use_sh_coeffs = False
        self.use_similarity_theory = False
        self.loss_weight = 1.0

        self.scale_loss_by_kernel_epsilon = False
        self.use_outpos_statistics = True

        self.outlier_distance = 4.0

        self.use_epsilon_space = False

    def __str__(self):
        return dict(self.__dict__).__str__()


import vae.config_abs
import vae.config_scatter

def get_config(module, config_name):

    if module == vae.config_abs and  config_name.lower() == 'randomabs':
        return vae.config_abs.generate_absorption_config()

    # list all classes in the current module: these are all possible configs
    configs = [obj for _, obj in inspect.getmembers(sys.modules[module.__name__]) if inspect.isclass(obj)]
    string_names = [c.__name__.lower() for c in configs]
    class_name = config_name.lower()
    if class_name in string_names:
        return configs[string_names.index(class_name)]()
    else:
        raise ValueError(f'Config {config_name} not found!')
