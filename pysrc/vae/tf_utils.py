from itertools import chain

import numpy as np
import tensorflow as tf

import vae.utils


def binned_average(values, selection_values, value_range, nBins, weights=None):
    """Computes an average of values binned according to the selection_values array
    Assumes the following shapes:
        values.shape  = (?, points, features)
        selection_values.shape = (?, points)

    returns the following shape: (?, nBins, features)

    """
    outputs = []
    all_counts = []
    indices = tf.histogram_fixed_width_bins(
        selection_values, value_range, nbins=nBins)
    for b in range(nBins):
        mask = tf.cast(tf.equal(indices, b), tf.float32)

        if weights is not None:
            mask = mask * weights
        mean = tf.reduce_sum(tf.expand_dims(mask, -1) * values, axis=1)
        counts = tf.reduce_sum(mask, axis=1, keepdims=True)
        outputs.append(mean / counts)
        all_counts.append(counts)

    counts = tf.stack(all_counts, axis=1)
    result = tf.stack(outputs, axis=1)
    result = tf.where(tf.is_nan(result), tf.zeros_like(result), result)

    return result, counts


def implicit_function(poly, query_positions, eps_val=0.02, use_cpp_impl=False):
    # if use_cpp_impl:
    #     return implicit_function_cpp(poly, query_positions, eps_val)
    print("No.of nodes: {}".format(len([n.name for n in tf.get_default_graph().as_graph_def().node])))

    # Evaluate function query points
    pos_p = tf.placeholder(tf.float32, [None, 2])

    def integrate_segment(p0_p, p1_p, pos_p):
        d = -p0_p + p1_p
        dLen = np.sqrt(np.sum(d ** 2, 1, keepdims=True))
        d = d / dLen

        # Project point onto segment
        a = -tf.reduce_sum((pos_p - p0_p) * d, 1, keepdims=True)
        b = -tf.reduce_sum((pos_p - p1_p) * d, 1, keepdims=True)
        proj = p0_p - a * d
        D = tf.sqrt(tf.reduce_sum((proj - pos_p) ** 2, 1, keepdims=True) + 0.0001)
        # Compute integral
        invDenom = 1 / tf.sqrt(D ** 2 + eps_val)
        integrated_weight = invDenom * (tf.atan(b * invDenom) - tf.atan(a * invDenom))
        return integrated_weight

    poly = poly.astype(np.float32)
    res_values = tf.zeros((query_positions.shape[0], 1))
    res_values_norm = tf.zeros_like(res_values)
    for i in range(poly.shape[0]):
        p0 = poly[i]
        p1 = poly[(i + 1) % poly.shape[0]]
        d = p1 - p0
        d = d / np.sqrt(np.sum(d ** 2))

        n = np.array([[-d[1], d[0]]])
        int_w = integrate_segment(p0[np.newaxis, :], p1[np.newaxis, :], pos_p)

        res_values += tf.reduce_sum((p0[np.newaxis, :] - pos_p) * n, 1, keepdims=True) * int_w
        res_values_norm += int_w
    values_op = res_values / res_values_norm
    values_grad_op = tf.gradients(values_op, pos_p)
    values_hessian_dx = tf.gradients(values_grad_op[0][:, 0], pos_p)
    values_hessian_dy = tf.gradients(values_grad_op[0][:, 1], pos_p)

    values_third_dx_dx = tf.gradients(values_hessian_dx[0][:, 0], pos_p)
    values_third_dy_dy = tf.gradients(values_hessian_dy[0][:, 1], pos_p)

    print("No.of nodes: {}".format(len([n.name for n in tf.get_default_graph().as_graph_def().node])))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # ret_val = sess.run([values_op, values_grad_op, values_hessian_dx, values_hessian_dy], feed_dict={pos_p: query_positions})
        ret_val = sess.run([values_op, values_grad_op, values_hessian_dx,
                            values_hessian_dy,
                            values_third_dx_dx, values_third_dy_dy], feed_dict={pos_p: query_positions})
        f_val = ret_val[0]
        grad_dx = ret_val[1][0][:, 0]
        grad_dy = ret_val[1][0][:, 1]
        hess_x = ret_val[2]
        hess_y = ret_val[3]
        hess_dx_dx = hess_x[0][:, 0]
        hess_dx_dy = hess_x[0][:, 1]
        hess_dy_dx = hess_y[0][:, 0]
        hess_dy_dy = hess_y[0][:, 1]

        third_dx_dx_dx = ret_val[4][0][:, 0]
        third_dx_dx_dy = ret_val[4][0][:, 1]

        third_dy_dy_dx = ret_val[5][0][:, 0]
        third_dy_dy_dy = ret_val[5][0][:, 1]

        return {'f': f_val,
                'dx': grad_dx,
                'dy': grad_dy,
                'dx_dx': hess_dx_dx,
                'dx_dy': hess_dx_dy,
                'dy_dy': hess_dy_dy,
                'dx_dx_dx': third_dx_dx_dx,
                'dx_dx_dy': third_dx_dx_dy,
                'dy_dy_dx': third_dy_dy_dx,
                'dy_dy_dy': third_dy_dy_dy}


def tf_basis_fun(ref_p, eval_p, poly_order):
    rel = eval_p - ref_p
    terms = []
    for i in range(poly_order + 1):
        for j in range(i + 1):
            tmp = (rel[:, :, 0] ** (i - j)) * (rel[:, :, 1] ** j)
            terms.append(tmp)
    result = tf.stack(terms, 2)
    return result


def tf_polynomial_to_voxel_grid(current_pos, current_normal, poly_coeffs, poly_order,
                                tangent_space, extent=[-1, 1, -1, 1], grid_res=64):

    if current_pos.shape[1] > 2:
        raise ValueError('Only supports 2D coordinates')
    # Evaluate taylor expansion
    x, y = np.meshgrid(np.linspace(extent[0], extent[1], grid_res), np.linspace(extent[2], extent[3], grid_res))
    coords = np.stack([x.ravel(), y.ravel()], 1)
    coords = coords[np.newaxis, :, :]
    if tangent_space:  # Convert to tangent space
        coords_ts = world_to_local(current_pos, current_normal, coords, True)
        b = tf_basis_fun(tf.expand_dims(current_pos, 1) * 0.0, coords_ts, poly_order)
    else:
        b = tf_basis_fun(tf.expand_dims(current_pos, 1), coords, poly_order)

    f_taylor = tf.reshape(tf.reduce_sum(b * tf.expand_dims(poly_coeffs, 1), axis=2),
                          [tf.shape(current_pos)[0], grid_res, grid_res])
    min_pos = np.array([[extent[0], extent[2]]])
    max_pos = np.array([[extent[1], extent[3]]])
    bb_diag = max_pos - min_pos
    center_coords = tf.cast(tf.reshape((grid_res * (current_pos - min_pos) / bb_diag), [-1, 2]), tf.int32)
    batch_idx = tf.range(start=0, limit=tf.shape(f_taylor)[0], delta=1)
    center_coords = tf.stack([batch_idx, center_coords[:, 1], center_coords[:, 0]], 1)
    fit_level = tf.gather_nd(f_taylor, center_coords)
    f_taylor -= tf.expand_dims(tf.expand_dims(fit_level, -1), -1)
    f_taylor = tf.expand_dims(f_taylor, -1)
    return f_taylor


def eval_poly(query_pos, ref_pos, ref_dir, poly_coeffs, poly_order, tangent_space, scale):
    """Evaluates a given polynomial. The code assumes batched input, in query_pos.shape = [None, 3]"""
    if tangent_space:
        pos = world_to_local(ref_pos, ref_dir, query_pos, True) * scale
    else:
        if ref_pos is not None:
            pos = (query_pos - ref_pos) * scale
        else:
            pos = query_pos * scale

    term_idx = 0

    if len(query_pos.shape) == 3:
        value = tf.zeros((tf.shape(query_pos)[0], query_pos.shape[1]), dtype=query_pos.dtype)
    else:
        value = tf.zeros((tf.shape(query_pos)[0]), dtype=query_pos.dtype)
    for d in range(poly_order + 1):
        for i in range(d + 1):
            for j in range(i + 1):
                dx = d - i
                dy = d - dx - j
                dz = d - dx - dy
                value += tf.expand_dims(poly_coeffs[:, term_idx], 1) * \
                    pos[..., 0] ** dx * pos[..., 1] ** dy * pos[..., 2] ** dz
                term_idx += 1
    return tf.expand_dims(value, -1)


def tf_eval_poly_gradient(query_pos, ref_pos, ref_dir, poly_coeffs, tangent_space, scale=1.0, poly_order=None):
    if poly_order is None:
        poly_order = vae.utils.extract_poly_order_from_n_coeffs(poly_coeffs.shape[1])
    if tangent_space:
        pos = world_to_local(ref_pos, ref_dir, query_pos, True)
    else:
        if ref_pos is not None:
            pos = query_pos - ref_pos
        else:
            pos = query_pos

    pos *= scale

    # Transform poly coeffs by applying the deriv matrix
    dim = query_pos.shape[1]
    deriv_coeffs = []
    values = []
    for d in range(dim):
        deriv_mat = vae.utils.deriv_matrix(dim, poly_order, d).astype(np.float32)
        deriv_coeffs = tf.transpose(tf.matmul(deriv_mat, tf.transpose(poly_coeffs)))
        term_idx = 0
        value = tf.zeros((tf.shape(query_pos)[0]), dtype=query_pos.dtype)
        for d in range(poly_order + 1):
            for i in range(d + 1):
                for j in range(i + 1):
                    dx = d - i
                    dy = d - dx - j
                    dz = d - dx - dy
                    value += deriv_coeffs[:, term_idx] * pos[:, 0] ** dx * pos[:, 1] ** dy * pos[:, 2] ** dz
                    term_idx += 1

        values.append(value)

    deriv = tf.stack(values, axis=1)
    if tangent_space:
        t1, t2 = vae.tf_utils.onb_duff(ref_dir)
        deriv = t1 * tf.expand_dims(deriv[:, 0], -1) + \
            t2 * tf.expand_dims(deriv[:, 1], -1) + ref_dir * tf.expand_dims(deriv[:, 2], -1)

    return deriv


def get_num_trainable_variables():
    """Returns the number of trainable params in the current graph"""
    total_parameters = 0
    for variable in tf.trainable_variables():
        variable_parameters = 1
        for dim in variable.get_shape():
            variable_parameters *= dim.value
        total_parameters += variable_parameters
    return total_parameters


def albedo_to_effective_albedo(albedo):
    return -tf.log(1.0 - albedo * (1.0 - tf.exp(-8.0))) / 8.0


def onb_duff(n):
    """Constructs an orthonormal basis from a single (normalized) vector n"""
    sign = tf.cast(tf.greater(n[..., 2], 0.0), tf.float32) * 2 - 1
    a = -1.0 / (sign + n[..., 2])
    b = n[..., 0] * n[..., 1] * a
    b1 = tf.stack([1.0 + sign * n[..., 0] * n[..., 0] * a, sign * b, -sign * n[..., 0]], axis=-1)
    b2 = tf.stack([b, sign + n[..., 1] * n[..., 1] * a, -n[..., 1]], axis=-1)
    return b1, b2


def azimuth_transform(ref_dir, normal):
    s, t = onb_duff(normal)
    ts_transform = tf.stack([s, t, normal], axis=-1)

    # Transform: Tangent Space => (0, y, z) Space
    ts_ref_dir = tf.matmul(ts_transform, tf.expand_dims(ref_dir, -1), transpose_a=True)

    phi = tf.atan2(ts_ref_dir[..., 0, 0], ts_ref_dir[..., 1, 0])
    cos_phi = tf.cos(phi)
    sin_phi = tf.sin(phi)

    _0 = tf.zeros_like(phi)
    _1 = tf.ones_like(phi)
    rot_mat = tf.stack([cos_phi, -sin_phi, _0, sin_phi, cos_phi, _0, _0, _0, _1], axis=1)
    rot_mat = tf.reshape(rot_mat, [-1, 3, 3])
    new_ref_dir = tf.matmul(rot_mat, ts_ref_dir)

    new_ref_dir = new_ref_dir[..., 0]
    # Transform: (0, y, z) => Light Space
    s, t = onb_duff(new_ref_dir)
    ls_transform = tf.stack([s, t, new_ref_dir], axis=-1)
    full_transform = tf.matmul(tf.matmul(ls_transform, rot_mat, transpose_a=True), ts_transform, transpose_b=True)
    return full_transform


def world_to_local(in_pos, in_normal, out_pos_ws, predict_in_tangent_space):
    if in_pos.shape[-1] == 3:
        if predict_in_tangent_space:
            rel_out_pos = out_pos_ws - in_pos
            tangent1, tangent2 = onb_duff(in_normal)
            c_n = tf.reduce_sum(rel_out_pos * in_normal, axis=-1)
            c_t1 = tf.reduce_sum(rel_out_pos * tangent1, axis=-1)
            c_t2 = tf.reduce_sum(rel_out_pos * tangent2, axis=-1)
            rel_out_pos = tf.stack([c_t1, c_t2, c_n], axis=-1)
        else:
            rel_out_pos = out_pos_ws - in_pos
    else:
        if predict_in_tangent_space:
            rel_out_pos = out_pos_ws - in_pos
            tangent = tf.stack([-in_normal[..., 1], in_normal[..., 0]], axis=-1)
            c_n = tf.reduce_sum(rel_out_pos * in_normal, axis=-1)
            c_t = tf.reduce_sum(rel_out_pos * tangent, axis=-1)
            rel_out_pos = tf.stack([c_n, c_t], axis=-1)
        else:
            rel_out_pos = out_pos_ws - in_pos
    return rel_out_pos


def local_to_world(in_pos, in_normal, out_pos_ts, predict_in_tangent_space, name='local_to_world'):
    with tf.name_scope(name) as scope:
        if in_pos.shape[-1] == 3:
            if predict_in_tangent_space:
                tangent1, tangent2 = onb_duff(in_normal)
                out_pos_ws = tf.expand_dims(out_pos_ts[..., 0], -1) * tangent1 + \
                    tf.expand_dims(out_pos_ts[..., 1], -1) * tangent2 + \
                    tf.expand_dims(out_pos_ts[..., 2], -1) * in_normal
                out_pos_ws = tf.add(out_pos_ws, in_pos, name=scope)
            else:
                out_pos_ws = tf.add(out_pos_ts, in_pos, name=scope)
        else:
            if predict_in_tangent_space:
                tangent = tf.stack([-in_normal[..., 1], in_normal[..., 0]], axis=-1)
                out_pos_ws = tf.expand_dims(out_pos_ts[..., 0], -1) * in_normal + \
                    tf.expand_dims(out_pos_ts[..., 1], -1) * tangent
                out_pos_ws = tf.add(out_pos_ws, in_pos, name=scope)
            else:
                out_pos_ws = tf.add(out_pos_ts, in_pos, name=scope)

    return out_pos_ws


def to_ref_dir_coords(x, ref_dir):
    tangent1, tangent2 = onb_duff(ref_dir)
    c_t1 = tf.reduce_sum(x * tangent1, axis=-1)
    c_t2 = tf.reduce_sum(x * tangent2, axis=-1)
    c_n = tf.reduce_sum(x * ref_dir, axis=-1)
    return tf.stack([c_t1, c_t2, c_n], axis=-1)


def from_ref_dir_coords(x, ref_dir):
    tangent1, tangent2 = onb_duff(ref_dir)
    return tf.expand_dims(x[..., 0], -1) * tangent1 + \
        tf.expand_dims(x[..., 1], -1) * tangent2 + \
        tf.expand_dims(x[..., 2], -1) * ref_dir


def world_to_local_new(x, in_pos, in_dir, in_normal, local_space='TS'):
    rel_out_pos = x - in_pos
    if local_space == 'WS':
        return rel_out_pos
    elif local_space == 'LS':
        return to_ref_dir_coords(rel_out_pos, -in_dir)
    elif local_space == 'AS':
        transf = azimuth_transform(-in_dir, in_normal)
        return tf.matmul(transf, tf.expand_dims(rel_out_pos, -1))[..., 0]
    elif local_space == 'TS':
        return to_ref_dir_coords(rel_out_pos, in_normal)


def local_to_world_new(x, in_pos, in_dir, in_normal, local_space='TS', name='local_to_world'):
    with tf.name_scope(name) as scope:
        if local_space == 'LS':
            x = from_ref_dir_coords(x, -in_dir)
        elif local_space == 'AS':
            transf = azimuth_transform(-in_dir, in_normal)
            x = tf.matmul(transf, tf.expand_dims(x, -1), transpose_a=True)[..., 0]
        elif local_space == 'TS':
            x = from_ref_dir_coords(x, in_normal)
        return tf.add(x, in_pos, name=scope)


def get_placeholder(tensor, return_op=False):
    tensor_list = tf.contrib.framework.nest.flatten(tensor)

    def get_inputs(obj):
        if isinstance(obj, tf.Tensor):
            return obj.op.inputs
        elif isinstance(obj, tf.Operation):
            return obj.inputs
        else:
            raise TypeError(obj)

    todo = list(chain(*[list((t, i) for i in get_inputs(t)) for t in tensor_list]))
    visited = set()
    placeholders = list()
    counter = 0
    while len(todo) > 0:
        parent, tensor = todo.pop(0)
        if tensor in visited:
            continue
        visited.add(tensor)
        if tensor.op.type == 'Placeholder':
            placeholders.append(tensor)
            counter += 1
        todo.extend(list((tensor, i) for i in tensor.op.inputs))

    if return_op:
        placeholders = [p.op for p in placeholders]

    return placeholders


def restore_model(session, logdir, scope=None):
    if scope:
        var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
        saver = tf.train.Saver(var)
    else:
        saver = tf.train.Saver()

    saver.restore(session, tf.train.latest_checkpoint(logdir))


def extract_voxel_grid(trainer, output_dir):

    img_coords = tf.stack([trainer.in_pos_p[:, 1] / 2, trainer.in_pos_p[:, 0] / 2], 1)

    polygons_train = vae.utils.load_polygons(os.path.join(output_dir, 'scenes', 'train'))
    polygons_test = vae.utils.load_polygons(os.path.join(output_dir, 'scenes', 'test'))

    # Discretize all polygons
    discrete_polygons_train = []
    # poly_order = extract_poly_order_from_feat_name(config.shape_features_name)
    # ws_feat = 'mlsPoly{}'.format(poly_order)
    # feed_dict[trainer.shape_features_ws_p] = shape_features_ws
    # feed_dict[trainer.shape_features_ws_mean_p] = feature_statistics['{}_mean'.format(ws_feat)]
    # feed_dict[trainer.shape_features_ws_stdinv_p] = feature_statistics['{}_stdinv'.format(ws_feat)]
    for poly in polygons_train:
        discrete = vae.utils.discretize_polygon(poly, 32)
        discrete = np.pad(discrete, [16, 16], 'constant')
        discrete = discrete[:, :, np.newaxis]
        discrete_polygons_train.append(discrete)
    discrete_polygons_test = []
    for poly in polygons_test:
        discrete = vae.utils.discretize_polygon(poly, 32)
        discrete = np.pad(discrete, [16, 16], 'constant')
        discrete = discrete[:, :, np.newaxis]
        discrete_polygons_test.append(discrete)

    # Put all polygons into a big tensor
    discrete_polygons_train = np.stack(discrete_polygons_train).astype(np.float32)
    discrete_polygons_test = np.stack(discrete_polygons_test).astype(np.float32)

    discrete_polygons = tf.cond(tf.squeeze(tf.equal(trainer.dataset_name_p, tf.constant('train'))),
                                lambda: discrete_polygons_train, lambda: discrete_polygons_test)
    selected_polys = tf.gather(discrete_polygons,  trainer.polygon_idx[:, 0])

    # Apply some convolutional layers to the local_grid
    current = tf.image.extract_glimpse(selected_polys, np.array([32, 32]), img_coords)
    return current


def dump_pbtxt_config(output_names, model_outputs, config, filename, batch_size, class_name):
    feeds = get_placeholder(model_outputs)
    feeds = sorted(feeds, key=lambda f: f.name.split(':')[0])
    with open(filename + '.pbtxt', 'w') as f:
        for feed in feeds:
            f.write('feed {\n')
            f.write('id {{ node_name: "{}"}}\n'.format(feed.name.split(':')[0]))
            f.write('shape {\n')
            if not feed.shape.ndims:
                f.write('dim { size: 1 }\n')
            else:
                for d in feed.shape:
                    if d.value is None:
                        f.write('dim {{ size: {} }}\n'.format(batch_size))
                    else:
                        f.write('dim {{ size: {} }}\n'.format(d))
            f.write('}\n')
            f.write('}\n')
        for n in output_names:
            f.write('fetch {\n')
            f.write('id {{ node_name: "{}" }}\n'.format(n))
            f.write('}\n')

    with open(filename + '.inc', 'w') as f:
        for idx, feed in enumerate(feeds):
            f.write('float* {}_data(Vae::{} *graph) {{\n'.format(feed.name.split(':')[0], class_name))
            f.write('   return graph->arg{}_data();\n'.format(idx))
            f.write('}\n\n')
        f.write('int batchSize{} = {};'.format(class_name, batch_size))
        f.write('int numShapeFeats = {};'.format(
            vae.utils.shape_feat_name_to_num_coeff(config.shape_features_name, config.dim)))
        f.write('int nLatent = {};'.format(config.n_latent))
