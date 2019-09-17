#!/usr/bin/env python3

import argparse
import json
import os
import re
import pickle
import subprocess
import sys
import time
import traceback

import numpy as np
import tensorflow as tf

import utils.namegen
import vae.config
import vae.config_abs
import vae.config_angular
import vae.config_scatter
import vae.datahandler
import vae.datapipeline
import vae.predictors
import vae.trainer
from utils.experiments import dump_config, get_existing_net, load_config
from vae.global_config import DATADIR, OUTPUT2D, OUTPUT3D, RESOURCEDIR


def run(args):
    # If multiple configs are supposed to be trained, recursively launch script with just one of them each
    if args.config:
        args.config = args.config[0]

    if args.output == '':
        args.output = OUTPUT3D

    traindata_file = os.path.join(args.output, 'datasets', args.datasetconfig, 'traindata')

    scene_dir = os.path.join(args.output, 'scenes')

    if args.abs:
        tensorflow_logdir = os.path.join(args.output, 'models_abs')
    elif args.angular:
        tensorflow_logdir = os.path.join(args.output, 'models_angular')
    else:
        tensorflow_logdir = os.path.join(args.output, 'models')

    # Find the last net we trained just in case we need it to continue training or run the predictor
    os.makedirs(tensorflow_logdir, exist_ok=True)
    existing_net_log_dir = get_existing_net(tensorflow_logdir, args.netname)

    os.makedirs(args.output, exist_ok=True)

    if args.gentraindata:
        datadir = os.path.join(args.output, DATADIR)

        os.makedirs(datadir, exist_ok=True)
        if args.datasetfolder is None:
            if args.datasetname is None:
                dataset_folder = '{:04}_{}'.format(
                    utils.namegen.get_next_experiment_number(datadir), args.datasetconfig)
            else:
                dataset_folder = '{:04}_{}'.format(
                    utils.namegen.get_next_experiment_number(datadir), args.datasetname)
            dataset_folder = os.path.join(datadir, dataset_folder)
            # Dump some basic info
            os.makedirs(dataset_folder, exist_ok=True)

            scatter_data = vae.datapipeline.get_config(args.datasetconfig)
            scatter_data = scatter_data(scene_dir, RESOURCEDIR, dataset_folder)
            utils.experiments.dump_dataset_info(dataset_folder, args.datasetconfig, scatter_data)
        else:
            dataset_folder = args.datasetfolder

        if args.jobid is not None:
            vae.datahandler.generate_training_data(scene_dir, args.datasetconfig,
                                                   args.output, dataset_folder, args.processonly, n_threads=args.ncores, cluster_job=True,
                                                   job_id=args.jobid, n_jobs=args.njobs)
        else:
            t0 = time.time()
            vae.datahandler.generate_training_data(scene_dir, args.datasetconfig,
                                                   args.output, dataset_folder, args.processonly, n_threads=args.ncores)
            t1 = time.time()
            print('Generating Traindata took: {} s'.format(t1 - t0))

    if args.computeerror:
        vae.datahandler.evaluate_histogram_errors_3d(
            args.output, traindata_file, os.path.join(scene_dir, 'test'), args.evalnets)


    if args.winding:
        vae.datahandler.fix_polygon_winding(args.output)

    if args.dryrun:
        if args.abs:
            config = vae.config.get_config(vae.config_abs, args.config)
        elif args.angular:
            config = vae.config.get_config(vae.config_angular, args.config)
        else:
            config = vae.config.get_config(vae.config_scatter, args.config)
        return

    if args.train:
        # Split the config into different subconfigs if applicable
        configs = args.config.split('/')
        multiple_configs = len(configs) > 1

        if args.restoretrain:
            # If we want to restore the training, overwrite all other config options by the loaded data
            log_dir = existing_net_log_dir
            loaded_config = load_config(log_dir)
            args.__dict__ = loaded_config['args']
            args.__dict__['restoretrain'] = True
        else:
            if args.netname == '':
                if args.config == 'randomabs':
                    log_dir = os.path.join(
                        tensorflow_logdir, utils.namegen.generate_experiment_name(tensorflow_logdir))
                else:
                    log_dir = os.path.join(tensorflow_logdir, '{:04}_{}'.format(
                        utils.namegen.get_next_experiment_number(tensorflow_logdir), '_'.join(configs)))
            else:
                log_dir = os.path.join(tensorflow_logdir, '{:04}_{}'.format(
                    utils.namegen.get_next_experiment_number(tensorflow_logdir), args.netname))
            if os.path.isdir(log_dir):
                print('Error: Trying to overwrite existing training run!')
                quit()

        config_abs, config_angular = None, None
        if multiple_configs:
            config = vae.config.get_config(vae.config_scatter, configs[0])
            if configs[1] != '':
                config_abs = vae.config.get_config(vae.config_abs, configs[1])
            if len(configs) > 2 and configs[2] != '':
                config_angular = vae.config.get_config(vae.config_angular, configs[2])
            if (config_abs and config_abs.dataset != config.dataset) or (config_angular and config_angular.dataset != config.dataset):
                raise ValueError('Different configs must use the same dataset')
        else:
            if args.abs:
                config = vae.config.get_config(vae.config_abs, args.config)
            elif args.angular:
                config = vae.config.get_config(vae.config_angular, args.config)
            else:
                config = vae.config.get_config(vae.config_scatter, args.config)

        if not args.restoretrain:
            os.makedirs(log_dir, exist_ok=True)

        config.dim = 3


        # fix numpy and tf random seeds for reproducibility of the training
        np.random.seed(config.seed)
        tf.set_random_seed(17 * config.seed)

        # Get dataset from model config
        dataset = vae.datapipeline.get_config_from_metadata(config.dataset, args.output)
        dataset = dataset(scene_dir, RESOURCEDIR, os.path.join(args.output, DATADIR, config.dataset))

        config.datasetdir = config.dataset
        if not args.restoretrain:
            configs = [config]
            if config_angular:
                configs.append(config_angular)
            if config_abs:
                configs.append(config_abs)
            dump_config(log_dir, args, configs)

        predictor_abs, predictor_angular = None, None
        predictors = []
        ph_manager = vae.predictors.PlaceholderManager(dim=3)
        if multiple_configs:
            predictor = vae.predictors.ScatterPredictor(ph_manager, config, args.output)
            predictors.append(predictor)
            if config_abs:
                predictor_abs = vae.predictors.AbsorptionPredictor(ph_manager, config_abs, args.output)
                predictors.append(predictor_abs)
            if config_angular:
                predictor_angular = vae.predictors.AngularScatterPredictor(ph_manager, config_angular, args.output)
                predictors.append(predictor_angular)
        else:
            if args.abs:
                predictor = vae.predictors.AbsorptionPredictor(ph_manager, config, args.output)
            elif args.angular:
                predictor = vae.predictors.AngularScatterPredictor(ph_manager, config, args.output)
            else:
                predictor = vae.predictors.ScatterPredictor(ph_manager, config, args.output)
            predictors.append(predictor)

        feature_statistics = dataset.get_feature_statistics()

        # back up data statistics
        with open(os.path.join(log_dir, 'data_stats.pickle'), 'wb') as f:
            pickle.dump(feature_statistics, f)

        next_element_train, next_element_test = dataset.get_dataset_iterator(
            config.batch_size, config.num_epochs, config, feature_statistics, args.abs)
        predictor.dump_graph_info(os.path.join(log_dir, 'graph'))

        vae.trainer.train(next_element_train, next_element_test,
                            config.learningrate, feature_statistics,
                            log_dir, config, args.restoretrain, args.ncores, predictors)


def main(args):
    """Main entry point of the training and dataset generation script
    """
    cmd_string = __file__ + ' ' + ' '.join(args)
    parser = argparse.ArgumentParser(description='''Generates training data and/or runs training of SSS predictor''')
    parser.add_argument('--netname', default='')
    parser.add_argument('--config', nargs='*', default=[])
    parser.add_argument('--output', default='')
    parser.add_argument('--rendercmd', default=None)

    parser.add_argument('--restoretrain', help='Restores the training state',
                        action='store_true')

    parser.add_argument('--gentraindata', help='If set, the script regenerates the training data',
                        action='store_true')
    parser.add_argument('--processonly', help='If set, the script regenerates the statistics, tfrecord from the pickled training data',
                        action='store_true')
    parser.add_argument('--train', help='If set, the scenes are rendered (EXPENSIVE)',
                        action='store_true')

    parser.add_argument('--drawpolygons', help='Draw new training polygons',
                        action='store_true')
    parser.add_argument('--winding', help='If set, allows to inspect and correct winding of the training data',
                        action='store_true')
    parser.add_argument('--ncores', default=12, type=int)
    parser.add_argument('--jobid', default=None, type=int)
    parser.add_argument('--njobs', default=10, type=int)

    parser.add_argument('--ntrainsamples', default=100000, type=int)
    parser.add_argument('--ntrainscenes', default=200, type=int)
    parser.add_argument('--datasetconfig', default='default', type=str)
    parser.add_argument('--datasetfolder', default=None, type=str)
    parser.add_argument('--datasetname', default=None, type=str)

    parser.add_argument(
        '--dryrun', help='If set, only checks whether the passed configs actually exit', action='store_true')
    parser.add_argument('--computeerror', help='If set, compute error over the trained models',
                        action='store_true')
    parser.add_argument('--evalnets', default='', nargs='*')

    parser.add_argument('--abs', help='If set, we will train just the absorption part of the network',
                        action='store_true')
    parser.add_argument('--reportfailure', help='If set, exceptions are forwarded to slack (useful for cluster runs)',
                        action='store_true')

    parser.add_argument('--angular', help='If set, we will train just the angular prediction', action='store_true')
    args = parser.parse_args(args)

    try:
        run(args)
    except:
        print(f"Command: {cmd_string}")
        print(f"Error: {traceback.format_exc()}")

if __name__ == "__main__":
    main(sys.argv[1:])
