import heapq
import math
import random

import numpy as np
import scipy.sparse


def weighted_sampling_without_replacement(weights, m):
    """Samples m weighted samples https://stackoverflow.com/a/20548895"""
    elt = [(math.log(random.random()) / weights[i], i) for i in range(len(weights))]
    return [x[1] for x in heapq.nlargest(m, elt)]


def reshape_sparse(c, shape):
    """Reshape the sparse matrix `c`.

    Returns a coo_matrix with shape `shape`.

    Only works for matrices, not higher dimensional tensors.

    From: https://stackoverflow.com/a/16532553
    """
    if not hasattr(shape, '__len__') or len(shape) != 2:
        raise ValueError('`shape` must be a sequence of two integers')
    nrows, ncols = c.shape
    size = nrows * ncols
    new_size = shape[0] * shape[1]
    if new_size != size:
        raise ValueError('total size of new array must be unchanged')

    flat_indices = ncols * c.row + c.col
    new_row, new_col = divmod(flat_indices, shape[1])
    b = scipy.sparse.coo.coo_matrix((c.data, (new_row, new_col)), shape=shape)
    return b


def logit(p):
    return np.log(p) - np.log(1 - p)

def angles_to_vector(theta, phi):
    if len(theta.shape) == 2:
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)
        return np.array([x, y, z]).T

    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return np.array([x, y, z])


def vector_to_angles(p):
    if len(p.shape) == 2:
        theta = np.arccos(p[:, 2])
        phi = np.arctan2(p[:, 1], p[:, 0])
        return theta, phi
    theta = np.arccos(p[2])
    phi = np.arctan2(p[1], p[0])
    return theta, phi


def onb_duff(n):
    """Constructs an orthonormal basis from a single (normalized) vector n"""
    sign = np.copysign(1.0, n[..., 2])
    a = -1.0 / (sign + n[..., 2])
    b = n[..., 0] * n[..., 1] * a
    b1 = np.stack([1.0 + sign * n[..., 0] * n[..., 0] * a, sign * b, -sign * n[..., 0]], axis=-1)
    b2 = np.stack([b, sign + n[..., 1] * n[..., 1] * a, -n[..., 1]], axis=-1)
    return b1, b2


def coordinate_frame(n):
    """Constructs an orthonormal basis from a single (normalized) vector n"""
    return onb_duff(n)


def onb_duff_voxel_grid(n):
    """Constructs an orthonormal basis from a normalized vector n. Assumes the input is a voxel grid of normals"""

    # Reshape into 2D array
    original_dim = n.shape
    n = np.reshape(n, [-1, original_dim[-1]])

    sign = np.copysign(1.0, n[:, 2])
    a = -1.0 / (sign + n[:, 2])
    b = n[:, 0] * n[:, 1] * a
    b1 = np.stack([1.0 + sign * n[:, 0] * n[:, 0] * a, sign * b, -sign * n[:, 0]], axis=1)
    b2 = np.stack([b, sign + n[:, 1] * n[:, 1] * a, -n[:, 1]], axis=1)
    b1 = np.reshape(b1, original_dim)
    b2 = np.reshape(b2, original_dim)
    return b1, b2


def sampleCosineDistribution():
    p = squareToUniformDiskConcentric(np.random.rand(2))
    z = np.sqrt(1.0 - p[0] ** 2 - p[1] ** 2)
    if z == 0:
        z = 1e-10
    return np.array([p[0], p[1], z])


def squareToUniformDiskConcentric(uv):
    r1 = 2.0 * uv[0] - 1.0
    r2 = 2.0 * uv[1] - 1.0
    if r1 == 0 and r2 == 0:
        r = 0
        phi = 0
    elif r1 * r1 > r2 * r2:
        r = r1
        phi = (np.pi / 4.0) * (r2 / r1)
    else:
        r = r2
        phi = (np.pi / 2.0) - (r1 / r2) * (np.pi / 4.0)
    return np.array([r * np.cos(phi), r * np.sin(phi)])


def normalize(v):
    """Normalizes one or multiple input vectors v to length 1"""
    v2 = np.atleast_2d(v)
    v2 = v2 / np.sqrt(np.sum(v2 ** 2, axis=1, keepdims=True))
    return v2.reshape(v.shape)


def dot(v, w, keepdims=True):
    return np.sum(v * w, axis=-1, keepdims=keepdims)


def norm(v, keepdims=True):
    return np.sqrt(np.sum(v ** 2, axis=-1, keepdims=keepdims))


def cross(v, w):
    return np.stack([v[..., 1] * w[..., 2] - v[..., 2] * w[..., 1],
                     v[..., 2] * w[..., 0] - v[..., 0] * w[..., 2], 
                     v[..., 0] * w[..., 1] - v[..., 1] * w[..., 0]], axis=-1)



def grid_coordinates_3d(min_pos, max_pos, res):
    x = np.linspace(min_pos[0], max_pos[0], res)
    y = np.linspace(min_pos[1], max_pos[1], res)
    z = np.linspace(min_pos[2], max_pos[2], res)
    x, y, z = np.meshgrid(x, y, z, indexing='ij')
    coords = np.stack([x, y, z], axis=3)
    voxel_size = (max_pos - min_pos) / res
    # coords += voxel_size[None, None, None, :] / 2
    return coords


def azimuth_transform(ref_dir, normal):

    # Transform: World Space to Tangent Space
    s, t = onb_duff(normal)
    ts_transform = np.stack([s, t, normal], axis=-1)
    print(f"ts_transform.shape: {ts_transform.shape}")

    # Transform: Tangent Space => (0, y, z) Space
    ts_ref_dir = ts_transform @ ref_dir[..., None]
    phi = np.arctan2(ts_ref_dir[..., 0, 0], ts_ref_dir[..., 1, 0])
    rot_mat = np.array([[np.cos(phi), -np.sin(phi), 0], [np.sin(phi), np.cos(phi), 0], [0, 0, 1]])
    new_ref_dir = rot_mat @ ts_ref_dir
    # Transform: (0, y, z) => Light Space
    new_ref_dir = new_ref_dir.ravel()
    s, t = onb_duff(new_ref_dir.ravel())
    ls_transform = np.stack([s, t, new_ref_dir], axis=0)
    full_transform = ls_transform @ rot_mat @ ts_transform
    print(f"full_transform.shape: {full_transform.shape}")
    return full_transform
