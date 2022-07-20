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
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
# Specifically use CIFAR-10 for this example...
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
# torchvision NOT installed due to OSError: [Errno 28] No space left on device: '/home/eao21/miniconda33/envs/improv.lib.python3.9/site-packages/torchvision-0.13.0.dist-info'
# Mapped to location not in-memory: TMPDIR=/home/eao21/tmp pip install
# https://github.com/pypa/pip/issues/7745
# https://askubuntu.com/questions/1326304/cannot-install-pip-module-because-there-is-no-space-left-on-device
# from torchvision import datasets
import traceback

import logging; logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class PyTorchProcessor(Actor):
    # TODO: Update ALL docstrings
    # TODO: Clean commented sections
    # TODO: add GPU/CPU option as input...device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ''' Using PyTorch
    '''

    def __init__(self, *args, data=CIFAR10(root="./data", download=False), model_path=None, config_file=None):
        super().__init__(*args)
        logger.info(data, model_path)
        # Add print statements - demo data
        # Only for simulation, data = not acquired in real-time, no need to print...
        # Make f-statements/strings
        # print('Data: ', data_path)
        # print('Model: ', model_path)
        # print('config_file: ', config_file)
        self.param_file = config_file
        # Not necessary for real running...only for playing around w/different data + models
        # if data_path is None:
        #     logger.error('Must specify data path.')
        # else:
        #     self.data_path = data_path
        if model_path is None:
            logger.error('Must specify pre-trained model.')
        else:
            self.model_path = model_path

    def load_data(self, data_path=None)
        # Specify for data type...this is specifically for CIFAR demo data...
        # Depends on file type...working w/downloaded data from PyTorch torchvision for now
        
        self.data = data

        # NOTE: FOR ACTUAL MODELS TO BE INTEGRATED W/PYTORCH, WE MUST USE TORCH.JIT or something NOT DATALOADER!
        # NOTE: Time to load data - 
        # Images one-by-one
        # Images in batches
        # Create DataLoader
        # Example to lock data so different processes don't simultaneously download the data and cause data corruption, if running multiple processes...actors...that download/load dataset
        # from filelock import FileLock
        # with FileLock('./data.lock'):
        #     DataLoader(train_subset, batch_size=batch_size, shuffle=True)

        # self.train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        # self.test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)

    def setup(self, model=None, weights=None):
        ''' Setup PyTorch
        '''
        # Really only for demo...
        if model is not None:
            self.model = model
        else:
            print('Specify model.')

        logger.info('Running setup for '+self.name)

        # Init model object
        # Initialize model
        self.weights =
        self.model =

        # Set model to eval mode
        model.eval()

        # self.loadParams(param_file=self.param_file)
        # self.params = self.client.get('params_dict')

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
        ''' Run the processor on input data
        '''

    def load_model(self, model_path=None):
        # Initialize model
        self.model = torch.jit.load('model_scripted.pt')
        self.model.eval()

# Create dataloaders for model...
# TODO: input = output from download_data.py and batch_size
# (download_data.py input = specified PyTorch dataset); 
# batch_size = 64

# # Create data loaders.
# train_dataloader = DataLoader(training_data, batch_size=batch_size)
# test_dataloader = DataLoader(test_data, batch_size=batch_size)