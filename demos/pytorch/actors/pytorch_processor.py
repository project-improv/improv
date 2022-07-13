import time
import pickle
import json
import os
import numpy as np
from improv.store import Limbo, CannotGetObjectError, ObjectNotFoundError
from os.path import expanduser
from queue import Empty
from improv.actor import Actor, RunManager
import traceback
import torch
from torch import nn
from torch.utils.data import DataLoader
# torchvision NOT installed due to OSError: [Errno 28] No space left on device: '/home/eao21/miniconda33/envs/improv.lib.python3.9/site-packages/torchvision-0.13.0.dist-info'
# Mapped to location not in-memory: TMPDIR=/home/eao21/tmp pip install
# https://github.com/pypa/pip/issues/7745
# https://askubuntu.com/questions/1326304/cannot-install-pip-module-because-there-is-no-space-left-on-device
# from torchvision import datasets
# from torchvision.transforms import ToTensor
import traceback

import logging; logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class PyTorchProcessor(Actor):
    # TODO: Update ALL docstrings
    # TODO: add GPU/CPU option as input...device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ''' Using PyTorch
    '''

    def __init__(self, *args, data_path=None, config_file=None):
        super().__init__(*args)
        # Add logger.info?
        # Add print statement - data_path
        print('data ', data_path, 'config_file', config_file)
        self.param_file = config_file
        self.data_path = data_path

    def setup(self):
        ''' Setup PyTorch
        '''
                logger.info('Running setup for '+self.name)
        self.done = False
        # Saving model?
        self.saving = False

        self.loadParams(param_file=self.param_file)
        self.params = self.client.get('params_dict')

    # def loadParams(self, param_file=None):
    #     ''' Load parameters from file or 'defaults' into store
    #         TODO: accept user input from GUI
    #     '''
    #     cwd = os.getcwd()+'/'
    #     if param_file is not None:
    #         try:
    #             params_dict = json.load(open(param_file, 'r'))
    #             # Load other keys in params_dict
    #         except Exception as e:
    #             logger.exception('File cannot be loaded. {0}'.format(e))
    #     else:
    #         logger.exception('Need a config file for PyTorch model!')
    #     self.client.put(params_dict, 'params_dict')

    def run(self):
        ''' Run the processor on input data?
        '''

# Create dataloaders for model...
# TODO: input = output from download_data.py and batch_size
# (download_data.py input = specified PyTorch dataset); 
# batch_size = 64

# # Create data loaders.
# train_dataloader = DataLoader(training_data, batch_size=batch_size)
# test_dataloader = DataLoader(test_data, batch_size=batch_size)