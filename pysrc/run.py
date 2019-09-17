#!/usr/bin/env python3
"""File to generate data, train model and produce predictions in one shot"""

import r_gen_train_data
import r_train
import r_pred


# Generate the trainin data
r_gen_train_data.main([
    '--gentraindata', '--render', '--preptraindata',
    '--tfrecordname', 'groundtruth'
])

# Train the model
r_train.main([])

# Predict the scattering parameters for all scenes in the training set
r_pred.main([])
