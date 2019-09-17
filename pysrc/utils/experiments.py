"""Various utilities to manage different experiments"""

import datetime
import inspect
import json
import os
import subprocess
import time

import numpy as np



def convert_to_string(obj):
    if hasattr(obj, 'name'):
        return obj.name

    if callable(obj) or inspect.isclass(obj):
        return obj.__name__

    if type(obj) is np.ndarray:
        return obj.tolist()

    return obj


def dump_dataset_info(output_dir, config, dataset_config):
    metadata = {
        'name': os.path.split(output_dir)[-1],
        'time': datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'),
        'git': {
            'revision': subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip().decode('utf-8'),
            'diff': subprocess.check_output(['git', 'diff']).strip().decode('utf-8'),
        },
        'config': config
    }

    d = dict(dataset_config.__dict__)
    d['medium_param_generator'] = type(d['medium_param_generator']).__name__
    d['polyfit_config'] = dataset_config.get_polyfit_config_dict()

    for k in d.keys():
        d[k] = convert_to_string(d[k])
    metadata[f'datasetconfig'] = d

    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, sort_keys=False, indent=4, ensure_ascii=False)


def dump_config(log_dir, args, configs):
    """Dumps some elementary information about the current runs into a log directory"""

    if type(configs) != list:
        configs = [configs]

    training_metadata = {
        'name': os.path.split(log_dir)[-1],
        'starttime': datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'),
        'git': {
            'revision': subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip().decode('utf-8'),
            'diff': subprocess.check_output(['git', 'diff']).strip().decode('utf-8'),
        },
        'args': dict(vars(args)),
    }
    for i, c in enumerate(configs):
        d = dict(c.__dict__)
        for k in d.keys():
            d[k] = convert_to_string(d[k])
        training_metadata[f'config{i}'] = d

    with open(os.path.join(log_dir, 'training-metadata.json'), 'w') as f:
        json.dump(training_metadata, f, sort_keys=False, indent=4, ensure_ascii=False)


def load_config(log_dir):
    with open(os.path.join(log_dir, 'training-metadata.json'), 'r') as f:
        return json.load(f)


def get_existing_net(log_dir_parent, experiment_name):
    """Gets either the latest net which was trained or a net specified by the user"""
    if experiment_name == '':
        trained_nets = [os.path.join(log_dir_parent, d)
                        for d in os.listdir(log_dir_parent) if os.path.isdir(os.path.join(log_dir_parent, d))]
        if trained_nets:
            return max(trained_nets, key=os.path.getmtime)
        else:
            return None
    else:
        return os.path.join(log_dir_parent, experiment_name)
