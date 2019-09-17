import numpy as np
import os
import re

def random_name():
    adj = ['able', 'amazing', 'best', 'better', 'big', 'peaceful', 'certain', 'clear', 'different',
           'early', 'easy', 'economic', 'federal', 'free', 'full', 'good', 'great', 'tough', 'high',
           'human', 'important', 'blue', 'large', 'late', 'little', 'local', 'long', 'low', 'major', 'green', 'sweet', 'new', 'old', 'only', 'other',
           'red', 'possible', 'public', 'real', 'recent', 'right', 'small', 'social', 'special', 'tremendous',
           'strong', 'sure', 'true', 'white', 'whole', 'young', 'fake']

    nouns = ['area', 'book', 'apple', 'banana', 'case', 'flag', 'pear', 'country', 'day', 'eye', 'fact',
             'family', 'monkey', 'group', 'hand', 'home', 'job', 'life', 'lot', 'man', 'money', 'month',
             'mother', 'mister', 'night', 'number', 'part', 'flower', 'place', 'point', 'pig', 'program', 'question',
             'right', 'room', 'school', 'state', 'story', 'student', 'study', 'system', 'thing', 'time', 'water', 'way',
             'week', 'word', 'work', 'world', 'year']

    return adj[int(np.random.rand() * len(adj))] + '_' + nouns[int(np.random.rand() * len(nouns))]



def get_next_experiment_number(logdir, files=False):
    if not files:
        trained_nets = [d.split('_', 1) for d in os.listdir(logdir) if os.path.isdir(logdir + '/' + d)]
        existing_indices = [int(d[0]) for d in trained_nets]
    else:
        all_files = [d for d in os.listdir(logdir) if os.path.isfile(os.path.join(logdir, d)) and len(re.findall(r'\d+', d)) > 0]
        if len(all_files) == 0:
            return 0
        existing_indices = [int(re.findall(r'\d+', d)[0]) for d in os.listdir(logdir) if os.path.isfile(os.path.join(logdir, d)) and len(re.findall(r'\d+', d)) > 0]
    next_index = 0
    if existing_indices:
        max_index = max(existing_indices)
        next_index = max_index + 1
    return next_index

def generate_experiment_name(logdir):
    trained_nets = [d.split('_', 1) for d in os.listdir(logdir) if os.path.isdir(logdir + '/' + d)]
    trained_net_basenames = [d[1] for d in trained_nets]
    next_index = get_next_experiment_number(logdir)

    for i in range(10000):
        new_basename = random_name()
        new_filename = '{:04}_{}'.format(next_index, new_basename)
        if not new_basename in trained_net_basenames:
            return new_filename
    print('ERROR: Could not create a folder, all name combinations seem to be already in use!')
    quit()