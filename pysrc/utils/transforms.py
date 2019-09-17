import numpy as np
from utils.math import coordinate_frame, normalize, cross, dot

def azimuth_transform(ref_dir, normal):

    # Transform: World Space to Tangent Space
    ref_dir = np.atleast_2d(ref_dir)
    normal = np.atleast_2d(normal)

    ref_dir = normalize(ref_dir)
    normal = normalize(normal)

    s, t = coordinate_frame(normal)
    ts_transform = np.stack([s, t, normal], axis=-1)

    # Transform: Tangent Space => (0, y, z) Space
    ts_ref_dir = np.transpose(ts_transform, [0, 2, 1]) @ ref_dir[..., None]
    phi = np.arctan2(ts_ref_dir[..., 0, 0], ts_ref_dir[..., 1, 0])
    _0 = np.zeros_like(phi)
    _1 = np.ones_like(phi)

    rot_mat = np.stack([np.cos(phi), -np.sin(phi), _0, np.sin(phi), np.cos(phi), _0, _0, _0, _1], axis=1)
    rot_mat = np.reshape(rot_mat, [-1, 3, 3])
    new_ref_dir = rot_mat @ ts_ref_dir
    new_ref_dir = new_ref_dir[..., 0]

    # Transform: (0, y, z) => Light Space
    s, t = coordinate_frame(new_ref_dir)
    ls_transform = np.stack([s, t, new_ref_dir], axis=-1)
    full_transform = np.transpose(ls_transform, [0, 2, 1]) @ rot_mat @ np.transpose(ts_transform, [0, 2, 1])

    eps = 1e-7
    cond = (np.abs(ts_ref_dir[..., 0, 0]) < eps) * (np.abs(ts_ref_dir[..., 1, 0]) < eps)
    full_transform[cond, :, :] = np.transpose(ts_transform, [0, 2, 1])[cond, :, :]

    return full_transform


def to_azimuth_space(light_dir, normal):
    return to_azimuth_space_new(light_dir, normal)
    
    single_element = len(light_dir.shape) == 1
    light_dir = normalize(light_dir)
    normal = normalize(normal)
    transform = azimuth_transform(light_dir, normal)
    if single_element:
        return transform[0, :, :]
    return transform

def to_azimuth_space_new(light_dir, normal):
    single_element = len(light_dir.shape) == 1
    light_dir = normalize(light_dir)
    normal = normalize(normal)
    light_dir = np.atleast_2d(light_dir)
    normal = np.atleast_2d(normal)

    t1 = normalize(cross(normal, light_dir))
    t2 = normalize(cross(light_dir, t1))

    inv_cross = (np.abs(dot(normal, light_dir)) > 0.99999).ravel()
    t1_new, t2_new = coordinate_frame(light_dir)

    t1[inv_cross, ...] = t1_new[inv_cross, ...]
    t2[inv_cross, ...] = t2_new[inv_cross, ...]

    transform = np.stack([t1, t2, light_dir], axis=-1)
    transform = np.transpose(transform, [0, 2, 1])

    # transform = azimuth_transform(light_dir, normal)
    if single_element:
        return transform[0, :, :]
    return transform



def to_light_space(light_dir, normal):
    single_element = len(light_dir.shape) == 1
    light_dir = np.atleast_2d(light_dir)
    light_dir = normalize(light_dir)
    tangent1, tangent2 = coordinate_frame(light_dir)
    transform = np.stack([tangent1, tangent2, light_dir], 1)
    if single_element:
        return transform[0, :, :]
    return transform


def to_tangent_space(light_dir, normal):
    single_element = len(normal.shape) == 1
    normal = np.atleast_2d(normal)
    normal = normalize(normal)
    tangent1, tangent2 = coordinate_frame(normal)
    transform = np.stack([normal, tangent1, tangent2], 1)
    if single_element:
        return transform[0, :, :]
    return transform


def transform_to(points, ref_point, light_dir, normal, space):

    d = {'LS': to_light_space, 'TS': to_tangent_space, 'AS': to_azimuth_space}
    if space == 'WS':
        return points - ref_point
    rel_pos = points - ref_point
    transf = d[space](light_dir, normal)
    return (transf @ rel_pos[..., None])[..., 0], transf
