import time
import pickle
import json
import os
import numpy as np
from improv.store import Limbo, CannotGetObjectError, ObjectNotFoundError
from os.path import expanduser
from queue import Empty
from improv.actor import Actor, RunManager

import torch
# Specifically use CIFAR-10 for this example...
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor

from PIL import Image
import torchvision.transforms as transforms

import traceback

import logging; logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class PyTorchProcessor(Actor):
    # TODO: Update ALL docstrings
    # TODO: Clean commented sections
    # TODO: add GPU/CPU option as input...device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ''' Using PyTorch
    '''

    # def __init__(self, data_path=None, model_path=None, config_file=None):
    def __init__(self, data_path='~/improv/demos/pytorch/data/CIFAR10', model_path='~/improv/demos/pytorch/models/AlexNet-CIFAR10.pt', config_file=None):
        super().__init__(*args, data_path=data_path, model=torch.jit.load(model_path), config_file=config_file)
        logger.info(data_path, model_path)
        self.img_number = 0
        if model_path is None:
            logger.error("Must specify a pre-trained model path.")
        else:
            self.model_path = model_path

    def setup(self):
        ''' Prep data and init model
        Prep data = slow, load image
        '''
        logger.info('Loading model for ' + self.name)
        self.done = False
        # Necessary? See above init
        self.model = torch.jit.load(self.model_path)

    def run(self):
        ''' Run the processor continually on input data, e.g.,images
        '''

        # Done offline - iterate through online, only interested in time it takes to acquire data/get data "from store", process one sample (inference), put estimate into store
        self.get_in_time = []
        self.load_data_time = []
        # Done offline
        # self.load_model = []
        self.inference_time = []
        self.put_out_time = []
        self.total_times = []
        self.timestamp = []
        self.counter = []

        with RunManager(self.name, self.runProcess, self.setup, self.q_sig, self.q_comm) as rm:
            logger.info(rm)

        print('Processor broke, avg time per image: ', np.mean(self.total_times, axis=0))
        print('Processor got through ', self.img_number, ' images')
        if not os._exists('output'):
            try:
                os.makedirs('output')
            except:
                pass
        if not os._exists('output/timing'):
            try:
                os.makedirs('output/timing')
            except:
                pass

        np.savetxt('output/timing/process_image_time.txt', np.array(self.total_times))
        np.savetxt('output/timing/process_timestamp.txt', np.array(self.timestamp))

        # np.savetxt('output/timing/putAnalysis_time.txt', np.array(self.putAnalysis_time))
        np.savetxt('output/timing/putModelOutput_time.txt', np.array(self.putModelOutput_time))
        np.savetxt('output/timing/procImage_time.txt', np.array(self.procImage_time))

    def runProcess(self):
        ''' Run process. Runs once per sample (in this case, .jpg images).
        Output (estimates) go in DS - ID is image name.
        '''

        # Necessary?
        init = self.params['init_batch']
        # Meh
        img = self._checkImage()
        img = self._loadImg()

        if img is not None:
            t = time.time()
            self.done = False
            try:
                self.img = self.client.getID(img[0][str(self.img_number)])
                self.img = self._loadImg(self.img, self.img_number+init)
                t2 = time.time()
                self._runInference(self.img, self.model)
                t3 = time.time()
                self.putModelOutput()
                t4 = time.time()
                self.timestamp.append([time.time(), self.img_number])
            # Insert exceptions here...
            # except:
        self.img_number += 1
        self.total_times.append(time.time()-t)
        else:
            pass

    def putModelOutput(self):
        ''' Put output of model into data store
        '''
        t = time.time()
        ids = []
        ids.append([self.client.put(self.output, 'output'+str(self.img_number)), 'output'+str(self.img_number)])

        self.put(ids)

        # self.q_comm.put([self.img_number])

        self.putModelOutputTime.append([time.time()-t])


    # def _processData(self, data):
    #     ''' Processing on data
    #     '''
    #     # Run additional processing steps here, if necessary...
    #     # return data

    def _runInference(self, data, model):

        self.output = self.model(self.tensor)

    # def _loadImage(self, img):
    #     ''' Load data - here, .jpg image
    #     '''
    #     import torch
        
    #     img = Image.open(self.img)
    #     transform = transforms.Compose([transforms.PILToTensor()])
    #     self.tensor = transform(img)

    #     return img_tensor

    def _checkImages(self):
        ''' Check to see if we have images for processing
        '''
        try:
            res = self.q_in.get(timeout=0.0005)
            return res
        #TODO: additional error handling
        except Empty:
            # logger.info('No frames for processing')
            return None

class NaNDataException(Exception):
    pass
