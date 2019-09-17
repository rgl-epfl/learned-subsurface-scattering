"""Methods to handle triangle meshes using Mitsuba"""
# %%

import os

import numpy as np

import mitsuba
from mitsuba.core import *
from mitsuba.render import SceneHandler


def sample_mesh(mesh_file, n_samples, density=None, blue_noise_radius=None, resource_dir='./resources'):
    fileResolver = Thread.getThread().getFileResolver()
    fileResolver.appendPath(resource_dir)
    fileResolver.appendPath(os.path.split(mesh_file)[0])
    paramMap = StringMap()
    paramMap['meshfile'] = mesh_file
    scene = SceneHandler.loadScene(fileResolver.resolve(resource_dir + '/scene_shape_template.xml'), paramMap)
    scene.initialize()
    shape = scene.getShapes()[0]
    if density is not None:
        n_samples = max(int(shape.getSurfaceArea() * density), 32)

    use_blue_noise = blue_noise_radius is not None
    if not use_blue_noise:
        blue_noise_radius = 10
    sampled_p, sampled_n = scene.sampleSurface(shape, n_samples, use_blue_noise, blue_noise_radius)
    n_samples = len(sampled_p) // 3
    return np.reshape(np.array(sampled_p), [n_samples, 3]), np.reshape(np.array(sampled_n), [n_samples, 3])
