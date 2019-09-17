#!/usr/bin/env python3

import argparse
import os
import subprocess
from shutil import copyfile


import tensorflow as tf
from google.protobuf import text_format

from vae import tf_utils
import vae.global_config
import utils.io

# The original freeze_graph function
# from tensorflow.python.tools.freeze_graph import freeze_graph

dir = os.path.dirname(os.path.realpath(__file__))
tensorflow_path = '../../tensorflow_1_8/'


def freeze_graph(model_dir, output_node_names, aot):
    """Extract the sub graph defined by the output nodes and convert 
    all its variables into constant 
    Args:
        model_dir: the root folder containing the checkpoint state file
        output_node_names: a string, containing all the output node's names, 
                            comma separated
    """

    if not tf.gfile.Exists(model_dir):
        raise AssertionError(
            "Export directory doesn't exists. Please specify an export "
            "directory: %s" % model_dir)

    if not output_node_names:
        print("You need to supply the name of a node to --output_node_names.")
        return -1

    checkpoint = tf.train.get_checkpoint_state(model_dir)
    input_checkpoint = checkpoint.model_checkpoint_path
    absolute_model_dir = "/".join(input_checkpoint.split('/')[:-1])
    output_graph = absolute_model_dir + "/frozen_model.pb"

    # We start a session using a temporary fresh Graph
    with tf.Session(graph=tf.Graph()) as sess:
        saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=True)
        saver.restore(sess, input_checkpoint)
        output_graph_def = tf.graph_util.convert_variables_to_constants(
            sess, tf.get_default_graph().as_graph_def(), output_node_names.split(","))

        # Finally we serialize and dump the output graph to the filesystem
        with tf.gfile.GFile(output_graph, "wb") as f:
            f.write(output_graph_def.SerializeToString())
        print("%d ops in the final graph." % len(output_graph_def.node))

        if aot:

            # tfcompile_path = '/hdd/code/tensorflow/bazel-bin/tensorflow/compiler/aot/tfcompile'
            # cmd = tfcompile_path + \
            #     ' --target_triple="gcc" --graph="{}" --config="your_graph_config.pbtxt" --entry_point="your_C_entry_point_name" --cpp_class="YourNamespace::YourCppClass" --out_object="output.lib" --out_header="output.h"'.format(
            #         output_graph)
            # print('cmd: {}'.format(cmd))
            # subprocess.call(cmd, shell=True)

            # optimize for inference
            names = output_node_names.split(",")
            tensors = [tf.get_default_graph().get_tensor_by_name("{}:0".format(n)) for n in names]
            feeds = tf_utils.get_placeholder(tuple(tensors))
            input_names = [n.name.split(':')[0] for n in feeds]
            input_names = ','.join(input_names)
            cmd = tensorflow_path + 'bazel-bin/tensorflow/python/tools/optimize_for_inference --input {0} --output {0} --frozen_graph=True --input_names={1} --output_names={2} '.format(
                output_graph, input_names, output_node_names)
            subprocess.call(cmd, shell=True)

            # Copy the required files into the right location
            resource_dir = './resources/xla/'
            xlapath = os.path.join(tensorflow_path, 'vaexla')
            os.makedirs(xlapath, exist_ok=True)

            # Copy: frozen graph, model.h, model.cpp, graph.inl, graph.pbtxt, BUILD to folder
            copyfile(os.path.join(resource_dir, 'model.h'), os.path.join(xlapath, 'model.h'))
            copyfile(os.path.join(resource_dir, 'model.cpp'), os.path.join(xlapath, 'model.cpp'))
            copyfile(os.path.join(resource_dir, 'BUILD'), os.path.join(xlapath, 'BUILD'))
            copyfile(output_graph, os.path.join(xlapath, 'frozen_model.pb'))
            copyfile(os.path.join(absolute_model_dir, 'graph.pbtxt'), os.path.join(xlapath, 'graph.pbtxt'))
            copyfile(os.path.join(absolute_model_dir, 'graph.inc'), os.path.join(xlapath, 'graph.inc'))

            # Run the build
            cwd = os.getcwd()
            os.chdir(tensorflow_path)
            subprocess.call('bazel build vaexla:xlavaesampler.so', shell=True)
            os.chdir(cwd)
            # After the build: Resulting library and header file need to be copied to dependency folder
            os.makedirs('../xla', exist_ok=True)
            copyfile(os.path.join(resource_dir, 'model.h'), os.path.join('../xla', 'model.h'))
            copyfile(os.path.join(tensorflow_path, 'bazel-bin', 'vaexla', 'xlavaesampler.so'),
                     os.path.join('../xla', 'libxlavaesampler.so'))
            copyfile(os.path.join(tensorflow_path, 'bazel-bin', 'vaexla', 'xlavaesampler.so'),
                     os.path.join('../xla', 'xlavaesampler.so'))


def replace(fname, word, replacement):
    with open(fname, 'r') as f:
        s = f.read()
        s = s.replace(word, replacement)
    with open(fname, 'w') as f:
        f.write(s)


def compile_xla(model_name):
    parts = model_name.split('/')
    model_name, abs_model_name, angular_model_name = None, None, None
    model_name = parts[0]
    if len(parts) > 1:
        abs_model_name = parts[1]
    if len(parts) > 2:
        angular_model_name = parts[2]
    joint_name = '_'.join(parts)

    print(f"model_name: {model_name}")
    model_path = os.path.join(args.outputdir, 'models', model_name)
    print(f"model_path: {model_path}")
    if abs_model_name:
        abs_model_path = os.path.join(args.outputdir, 'models_abs', abs_model_name)
    else:
        abs_model_path = model_path

    if not tf.gfile.Exists(model_path):
        raise AssertionError(
            "Export directory doesn't exists. Please specify an export "
            "directory: %s" % model_path)

    model_names = ['absorption', 'out_pos_gen', 'out_pos_gen_batched']
    output_names = ['absorption/absorption', 'VAE/out_pos_gen', 'VAE/out_pos_gen']
    class_names = ['Absorption', 'OutPosGen', 'OutPosGenBatched']
    model_paths = [abs_model_path, model_path, model_path]

    for i in range(len(model_names)):
        model_name = model_names[i]
        class_name = class_names[i]
        output_name = output_names[i]
        
        print(f"model_paths[i]: {model_paths[i]}")
        checkpoint = tf.train.get_checkpoint_state(model_paths[i])
        input_checkpoint = checkpoint.model_checkpoint_path
        absolute_model_dir = "/".join(input_checkpoint.split('/')[:-1])
        output_graph = absolute_model_dir + "/frozen_model.pb"

        with tf.Session(graph=tf.Graph()) as sess:
            saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=True)
            saver.restore(sess, input_checkpoint)
            all_nodes = [n.name for n in tf.get_default_graph().as_graph_def().node]
            print(f"all_nodes: {all_nodes}")

            output_graph_def = tf.graph_util.convert_variables_to_constants(
                sess,  tf.get_default_graph().as_graph_def(), [output_name])

            with tf.gfile.GFile(output_graph, "wb") as f:
                f.write(output_graph_def.SerializeToString())
            print("%d ops in the final graph." % len(output_graph_def.node))

            # Copy the required files into the right location
            resource_dir = './resources/xla/'
            target_name = 'vaexla_{}'.format(model_name)
            xlapath = os.path.join(tensorflow_path, target_name)
            os.makedirs(xlapath, exist_ok=True)
            hdr = os.path.join(xlapath, 'model_{}.h'.format(model_name))
            src = os.path.join(xlapath, 'model_{}.cpp'.format(model_name))
            if model_name == 'absorption':
                copyfile(os.path.join(resource_dir, 'model_absorption.h'), hdr)
                copyfile(os.path.join(resource_dir, 'model_absorption.cpp'), src)
            else:
                copyfile(os.path.join(resource_dir, 'model.h'), hdr)
                copyfile(os.path.join(resource_dir, 'model.cpp'), src)
            build_file = os.path.join(xlapath, 'BUILD')
            copyfile(os.path.join(resource_dir, 'BUILD'), build_file)

            model_name2 = model_name[:1].upper() + model_name[1:]

            replace(hdr, '%MODELNAME%', model_name2)
            replace(hdr, '%CLASSNAME%', class_name)
            replace(src, '%MODELNAME%', model_name2)
            replace(src, '%LMODELNAME%', model_name)
            replace(src, '%CLASSNAME%', class_name)
            replace(src, '%LCLASSNAME%', class_name.lower())
            replace(build_file, '%MODELNAME%', model_name)
            replace(build_file, '%CLASSNAME%', class_name)

            copyfile(output_graph, os.path.join(xlapath, 'frozen_model.pb'))
            copyfile(os.path.join(absolute_model_dir, 'graph_{}.pbtxt'.format(
                model_name)), os.path.join(xlapath, 'graph.pbtxt'))
            copyfile(os.path.join(absolute_model_dir, 'graph_{}.inc'.format(
                model_name)), os.path.join(xlapath, 'graph.inc'))
            cwd = os.getcwd()
            os.chdir(tensorflow_path)
            compile_cmd = 'bazel build {}:{}.so'.format(target_name, 'xla' + model_name)
            print('compile_cmd: {}'.format(compile_cmd))
            subprocess.call(compile_cmd, shell=True)
            print('done build')
            os.chdir(cwd)
            os.makedirs('../xla', exist_ok=True)
            copyfile(hdr, os.path.join('../xla', 'model_{}.h'.format(model_name)))
            copyfile(os.path.join(tensorflow_path, 'bazel-bin', target_name, 'xla' + model_name + '.so'),
                     os.path.join('../xla', 'lib' + 'xla' + model_name + '.so'))
            copyfile(os.path.join(tensorflow_path, 'bazel-bin', target_name, 'xla' + model_name + '.so'),
                     os.path.join('../xla', 'xla' + model_name + '.so'))

    output_names = ['VAE/absorption', 'VAE/out_pos_gen']

    # Save the graph with all outputs in the current folder as frozen graph to be loaded by mitsuba
    # => This is done automatically from the render script
    # with tf.Session(graph=tf.Graph()) as sess:
    #     saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=True)
    #     saver.restore(sess, input_checkpoint)

    #     output_graph_def = tf.graph_util.convert_variables_to_constants(
    #         sess,  tf.get_default_graph().as_graph_def(), output_names)

    #     with tf.gfile.GFile(output_graph, "wb") as f:
    #         f.write(output_graph_def.SerializeToString())
    #     print("%d ops in the final graph." % len(output_graph_def.node))

    #     # optimize for inference
    #     names = output_names
    #     tensors = [tf.get_default_graph().get_tensor_by_name("{}:0".format(n)) for n in names]
    #     feeds = tf_utils.get_placeholder(tuple(tensors))
    #     input_names = [n.name.split(':')[0] for n in feeds]
    #     input_names = ','.join(input_names)

    #     cwd = os.getcwd()
    #     # NOTE: This could fail if the wrong tensorflow binary is in the path (e.g. added by using  export LD_LIBRARY_PATH=/usr/local/lib/:$LD_LIBRARY_PATH)
    #     cmd = 'python -m tensorflow.python.tools.optimize_for_inference --input {0} --output {0} --frozen_graph=True --input_names={1} --output_names={2} '.format(
    #         output_graph, input_names, ','.join(output_names))
    #     subprocess.call(cmd, shell=True)


def simple_freeze(model_dir, output_names=['absorption/absorption', 'scatter/out_pos_gen', 'angular/out_dir_gen'], dump_variables=True):
    if not tf.gfile.Exists(model_dir):
        raise AssertionError(
            "Export directory doesn't exists. Please specify an export "
            "directory: %s" % model_dir)

    checkpoint = tf.train.get_checkpoint_state(model_dir)
    input_checkpoint = checkpoint.model_checkpoint_path
    absolute_model_dir = "/".join(input_checkpoint.split('/')[:-1])
    output_graph = absolute_model_dir + "/frozen_model.pb"

    variable_folder = os.path.join(absolute_model_dir, 'variables')
    os.makedirs(variable_folder, exist_ok=True)
    # Save the graph with all outputs in the current folder as frozen graph to be loaded by mitsuba
    with tf.Session(graph=tf.Graph()) as sess:
        saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=True)
        saver.restore(sess, input_checkpoint)

        output_graph_def = tf.graph_util.convert_variables_to_constants(
            sess, tf.get_default_graph().as_graph_def(), output_names)

        with tf.gfile.GFile(output_graph, "wb") as f:
            f.write(output_graph_def.SerializeToString())
        print("%d ops in the final graph." % len(output_graph_def.node))

        # optimize for inference
        if False:  # TODO: We are currently not using optimize_for_inference at it seems to sometimes produce invalid graphs...
            names = output_names
            tensors = [tf.get_default_graph().get_tensor_by_name("{}:0".format(n)) for n in names]
            feeds = tf_utils.get_placeholder(tuple(tensors))
            input_names = [n.name.split(':')[0] for n in feeds]
            input_names = ','.join(input_names)

            cwd = os.getcwd()
            # NOTE: This could fail if the wrong tensorflow binary is in the path (e.g. added by using  export LD_LIBRARY_PATH=/usr/local/lib/:$LD_LIBRARY_PATH)
            cmd = 'python -m tensorflow.python.tools.optimize_for_inference --input {0} --output {0} --frozen_graph=True --input_names={1} --output_names={2} '.format(
                output_graph, input_names, ','.join(output_names))
            subprocess.call(cmd, shell=True)

        if dump_variables:
            # Write all relevant variables to disk
            variables_names = [v.name for v in tf.trainable_variables()]
            values = sess.run(variables_names)
            for k, v in zip(variables_names, values):
                v = v.T
                print(f"Shape: {v.shape}")
                file_name = k.split(':')[0].replace('/', '_')
                # if file_name == 'absorption_mlp_fcn_0_weights':
                #     print(f"v: {v}")
                #     print(v[63, 11]) 
                #     print(v[63, 12]) 
                #     print(v[63, 13]) 
                #     print(v[63, 14]) 
                #     print(v[63, 15]) 
                #     print(v[63, 16]) 
                #     print(v[63, 17]) 
                #     print(v[63, 18]) 
                #     print(v[63, 19]) 
                #     print(v[63, 20]) 
                #     print(v[63, 21]) 
                #     print(v[63, 22]) 

                save_path = os.path.join(variable_folder, file_name + '.bin')
                utils.io.save_np_to_file(save_path, v)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--outputdir', default=vae.global_config.OUTPUT3D)
    parser.add_argument("--model_dir", type=str,
                        default="/hdd/code/mitsuba-ml/pysrc/outputs/vae3d/models/0015_mlp_shape_features_deg_ls3", help="Model folder to export")
    parser.add_argument("--model", type=str, default='0270_VaeFeaturePre')
    parser.add_argument("--output_node_names", type=str, default="VAE/out_pos_gen,VAE/absorption",
                        help="The name of the output nodes, comma separated.")
    parser.add_argument('--freezeonly', help='If set, just freeze the graph without compiling XLA', action='store_true')

    aot = True
    args = parser.parse_args()

    # freeze_graph(args.model_dir, args.output_node_names, aot)
    if args.freezeonly:
        simple_freeze(args.model_dir)
    else:
        compile_xla(args.model)
