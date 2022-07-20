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
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor

from PIL import Image

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
    def __init__(self, *args, data_path='~/improv/demos/pytorch/data/CIFAR10', model_path='~/improv/demos/pytorch/models/AlexNet-CIFAR10.pt', config_file='~/improv/demos/pytorch/pytorch_demo.yaml', device=None):
        super().__init__(*args, data_path=data_path, model_path=model_path, config_file=config_file)
        logger.info(data_path, model_path)
        self.img_number = 0
        if model_path is None:
            logger.error("Must specify a pre-trained model path.")
        else:
            self.model_path = model_path
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def setup(self):
        ''' Init model
        Prep data = slow, load image
        Done in runAcquirer, acquire_folder, q._in - _checkImage
        Process done in _loadImage
        '''
        logger.info('Loading model for ' + self.name)
        self.done = False
        # Necessary? See above init
        # Load model offline, test how long it takes to load model - do outside processor - fine at init setup
        t = time.time()
        self.model = torch.jit.load(self.model_path).to(self.device)
        self.load_model_time = time.time() - t
        # np.savetxt('output/timing/load_model_time.txt', np.array(self.load_model_time))

    def run(self):
        ''' Run the processor continually on input data, e.g.,images
        '''

        # Done offline - iterate through online, only interested in time it takes to acquire data/get data "from store", process one sample (inference), put estimate into store
        self.load_model_time = [] # Done offline/in setup
        self.get_in_time = [] # acquire time
        self.load_img_time = []
        self.inference_time = []
        self.put_out_time = []
        self.total_times = []
        self.timestamp = []
        self.counter = []

        with RunManager(self.name, self.runProcess, self.setup, self.q_sig, self.q_comm) as rm:
            logger.info(rm)

        print('Processor broke, avg time per image: ', np.mean(self.total_times, axis=0))
        print('Processor got through ', self.img_number, ' images')
        out_path = '~/improv/demos/pytorch/output'
        if not os._exists(out_path):
            try:
                os.makedirs(out_path)
            except:
                pass
        if not os._exists(out_path):
            try:
                os.makedirs(out_path)
            except:
                pass

        np.savetxt(out_path + '/timing/total_times.txt', np.array(self.total_times))
        np.savetxt(out_path + '/timing/process_timestamp.txt', np.array(self.timestamp))

        np.savetxt(out_path +'/timing/load_model_time.txt', np.array(self.load_model_time))
        np.savetxt(out_path +'/timing/get_in_time.txt', np.array(self.get_in_time)) # acquire time...same
        np.savetxt(out_path +'/timing/put_out_time.txt', np.array(self.put_out_time))
        np.savetxt(out_path +'/timing/load_img_time.txt', np.array(self.load_img_time))
        np.savetxt(out_path +'/timing/inference_time.txt', np.array(self.inference_time))


    def runProcess(self):
        ''' Run process. Runs once per sample (in this case, .jpg images).
        Output (estimates) go in DS - ID is image name.
        '''

        self.img = self._checkImage()

        if img is not None:
            t = time.time()
            self.done = False
            try:
                # self.img = self.client.getID(img[0][str(self.img_number)])
                self.img = self._loadImg(self.img, self.img_number)
                t2 = time.time()
                self._runInference(self.img.to(device), self.model.to(device))
                t3 = time.time()
                self.putModelOutput()
                t4 = time.time()
                self.timestamp.append([time.time(), self.img_number])
            # Insert exceptions here...ERROR HANDLING, SEE ANNE'S ACTORS - from 1p demo
            except ObjectNotFoundError:
                logger.error('Processor: Image {} unavailable from store, droppping'.format(self.img_number))
                self.dropped_img.append(self.img_number)
                self.q_out.put([1])
            except KeyError as e:
                logger.error('Processor: Key error... {0}'.format(e))
                # Proceed at all costs
                self.dropped_img.append(self.img_number)
            except Exception as e:
                logger.error('Processor error: {}: {} during image number {}'.format(type(e).__name__,
                                                                                            e, self.frame_number))
                print(traceback.format_exc())
                self.dropped_img.append(self.img_number)
            self.img_number += 1
            self.total_times.append(time.time()-t)
            self.load_img_time = t2 - t
            self.inference_time = t2 - t3
            self.put_out_time = t4 - t3
        else:
            pass
        
    def putModelOutput(self):
        ''' Put output of model into data store
        '''
        # t = time.time()
        ids = []
        ids.append([self.client.put(self.output, 'output'+str(self.img_number)), 'output'+str(self.img_number)])

        self.put(ids)

        self.q_comm.put([self.img_number])

        # self.put_out_time.append([time.time()-t])

    # SEE _loadImage
    # def _processData(self, data):
    #     ''' Processing on data
    #     '''
    #     # Run additional processing steps here, if necessary...
    #     # return data

    def _runInference(self, data, model):

        data = data.to(self.device)
        self.output = self.model(data).to(device)

    def _loadImage(self, img):
        ''' Load data - here, .jpg image to tensor
        TODO: time
        '''
        # t = time.time()
        if img is None:
            raise ObjectNotFoundError
        img = Image.open(self.img)
        transform = transforms.Compose([transforms.PILToTensor()])
        img_tensor = transform(img)

        # self.load_img_time.append([time.time() - t])

        return img_tensor

    def _checkImages(self):
        ''' Check to see if we have images for processing
        From basic demo
        '''
        t = time.time()
        try:
            res = self.q_in.get(timeout=0.0005)
            return res
        #TODO: additional error handling
        except Empty:
            logger.info('No images for processing')
            return None
        self.get_in_time = time.time() - t

class NaNDataException(Exception):
    pass
