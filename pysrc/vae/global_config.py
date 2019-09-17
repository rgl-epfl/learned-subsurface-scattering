import os

OUTPUT2D = './outputs/vae2d/'
OUTPUT3D = './outputs/vae3d/'

RESOURCEDIR = './resources'
DATADIR = 'datasets/'
SCENEDIR = 'scenes/'

DATADIR3D = os.path.join(OUTPUT3D, DATADIR)
SCENEDIR3D = os.path.join(OUTPUT3D, SCENEDIR)
RENDERPATH = os.path.join(OUTPUT3D, 'render')
FIGUREPATH = os.path.join(OUTPUT3D, 'figures')
REPORTDIR = os.path.join(OUTPUT3D, 'report')

POINTDENSITY = 2
FIT_KERNEL_EPSILON = 20
FIT_REGULARIZATION = 1e-2
FIT_REGULARIZATION = 0.0001
FIT_KDTREE_THRESHOLD = 0.0
IGNORE_ZERO_SCATTER = True

MAX_G = 0.95
