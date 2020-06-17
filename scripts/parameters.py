from __future__ import print_function # added this to support usage of python2

import os
import numpy as np
from enum import Enum
import pickle

class MemUpdateStrategy(Enum):
    HIGH_LEARNING_PROGRESS = 0
    LOW_LEARNING_PROGRESS = 1
    RANDOM = 2


class Parameters:

    def __init__(self):
        self.dictionary = {
            'model_type':'mlp', # recurrent or mlp
            'directory': '',
            'normalization_limits': [-0.9, 0.9],
            'day_size': 288, # how many samples in a day
            'days_in_window': 5,  # how many days in the window
            'step': 10,  # was 2
            'max_epochs': 0,
            'loss': 'mean_squared_error',
            'optimizer': 'adam',
            'inp_sensors': ['temp', 'rh', 'co2', 'rad', 'leaft1', 'leaft2'],
            'out_sensors': ['trans', 'photo'],
            'arx': False,
            'adaptive': True,
            'memory_size': 1000,
            'memory_update_probability': 0.0001,
            'memory_update_strategy': MemUpdateStrategy.RANDOM,  # possible choices:  random, learning_progress
            'batch_size': 32,
            'batchs_to_update_online': 3,
            'mse_test_dataset_fraction' : 20,  #   how many samples to use in the MSE calculations? dataset_size / this.
            'mse_calculation_step': 4, # calculate MSE every X model fits
            'experiment_repetition': -1,
            'verbosity_level': 1
        }

        #with open(os.path.join( self.get('directory'), 'parameters.txt'), 'w') as f:
        #    print(self.dictionary, file = f)

    def get_window_size(self):
        return int(self.dictionary['day_size'] * self.dictionary['days_in_window'])

    def get(self, key_name):
        if key_name in self.dictionary.keys():
            return self.dictionary[key_name]
        else:
            print('Trying to access parameters key: '+ key_name+ ' which does not exist')

    def set(self, key_name, key_value):
        if key_name in self.dictionary.keys():
            print('Setting parameters key: ', key_name, ' to ', str(key_value))
            self.dictionary[key_name] = key_value
        else:
            print('Trying to modify parameters key: '+ key_name+ ' which does not exist')

    def save(self):
        # save as numpy array
        #np.save(os.path.join(self.get('directory'), 'parameters'), self.dictionary)
        pickle.dump(self.dictionary, open(os.path.join(self.get('directory'), 'parameters.pkl'), 'wb'),  protocol=2) # protcolo2 for compatibility with python2
        # save also as plain text file
        with open(os.path.join(self.get('directory'), 'parameters.txt'), 'w') as f:
            print(self.dictionary, file=f)