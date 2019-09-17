#%%
"""Different configurations for neural network to run"""

import model
import global_config


class ConfigBase:
    def __init__(self):
        self.patchsize = 16
        self.numepochs = 10000
        self.batchsize = 8
        self.seed = global_config.SEED
        self.learningrate = 5e-5
        self.model = None
        self.use_tangent_coordinates = False


class BaselinePointlight(ConfigBase):

    def __init__(self):
        super(BaselinePointlight, self).__init__()
        self.model = model.baseline_pointlight


class SogPointlight(ConfigBase):

    def __init__(self):
        super(SogPointlight, self).__init__()
        self.model = model.sum_of_gaussian_pointlight


class SogPointlightWide(ConfigBase):

    def __init__(self):
        super(SogPointlight, self).__init__()
        self.model = model.sum_of_gaussian_pointlight_wide


class ProjectedGaussian(ConfigBase):

    def __init__(self):
        super(ProjectedGaussian, self).__init__()
        self.model = model.baseline_gaussian


def get_config(config_name):
    configs = [BaselinePointlight, SogPointlight, SogPointlightWide, ProjectedGaussian]
    string_names = [c.__name__ for c in configs]
    class_name = str.join('', [part.capitalize() for part in config_name.split('_')])

    if class_name in string_names:
        return configs[string_names.index(class_name)]()

    else:
        print('Config {} not found!'.format(config_name))
        quit()