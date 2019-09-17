#!/usr/bin/env python3

import argparse 
import os 
import sys 
import glob
import shutil

import freeze_model
import json

args = sys.argv[1:]


def load_config(file):
    with open(file, 'r') as f:
        return json.load(f)


cmd_string = __file__ + ' ' + ' '.join(args)
parser = argparse.ArgumentParser(description='''Extracts all weights for rendering''')
parser.add_argument('--netname', default='0003_VaeFeaturePreSharedSim2_AbsSharedSimComplex')
parser.add_argument('--outputdir', default='./outputs/vae3d/')
args = parser.parse_args(args)


trained_models = sorted(glob.glob(os.path.join(args.outputdir, 'models', '*')))
trained_models = [os.path.split(s)[-1] for s in trained_models]

t = args.netname

parts = t.split('/')
t_model, t_abs_model, t_angular_model = None, None, None
t_model = parts[0]
if len(parts) > 1:
    t_abs_model = parts[1]
if len(parts) > 2:
    t_angular_model = parts[2]
t = '_'.join(parts)

model, model_abs, model_angular = None, None, None

package_dir = os.path.join(args.outputdir, 'package', 'outputs')
os.makedirs(package_dir, exist_ok=True)

if t_model in trained_models:
    model = t_model
    model_abs = t_abs_model

    model_angular = t_angular_model
    
    model_dir = os.path.join(args.outputdir, 'models', model)
    freeze_model.simple_freeze(model_dir, ['scatter/out_pos_gen'])
    variable_path = os.path.join(model_dir, 'variables')
    metadata_file = os.path.join(model_dir, 'training-metadata.json')

    output_variable_path = os.path.join(package_dir, 'vae3d', 'models', model, 'variables')
    
    
    shutil.copytree(variable_path, output_variable_path)
    shutil.copy(metadata_file, os.path.join(package_dir, 'vae3d', 'models', model, 'training-metadata.json'))

    if model_abs:
        abs_model_dir = os.path.join(args.outputdir, 'models_abs', model_abs)
        freeze_model.simple_freeze(abs_model_dir, ['absorption/absorption'])

    # Copy the dataset 
    config = load_config(metadata_file)['config0']

    data = config['dataset']
    dataset_dir = os.path.join(args.outputdir, 'datasets', data)
    dataset_dir_target = os.path.join(package_dir, 'vae3d', 'datasets', data)
    os.makedirs(dataset_dir_target, exist_ok=True)
    os.makedirs(os.path.join(dataset_dir_target, 'train'), exist_ok=True)
    os.makedirs(os.path.join(dataset_dir_target, 'test'), exist_ok=True)
    to_copy = ['metadata.json', 'train/data_stats.json', 'train/data_stats.pickle', 'test/data_stats.json', 'test/data_stats.pickle']
    for f in to_copy:
        shutil.copy(os.path.join(dataset_dir, f), os.path.join(dataset_dir_target, f))





# Copy the resulting variables 


# Copy the dataset statistics 
