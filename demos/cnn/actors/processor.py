import os
import time
import numpy as np
import pandas as pd
from improv.store import CannotGetObjectError, ObjectNotFoundError
from queue import Empty
from improv.actor import Actor, RunManager

import torch
from scipy.special import softmax

import logging; logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import traceback

class CNNProcessor(Actor):
    # TODO: Update ALL docstrings
    # TODO: Clean commented sections
    # TODO: Add any relevant q_comm
    ''' 
    '''

    def __init__(self, *args, n_imgs=None, gpu=False, gpu_num=None, model_path=None, classify=False, labels=None, out_path=None, method='spawn', **kwargs):
        super().__init__(*args, **kwargs)
        logger.info(model_path)
        if model_path is None:
            # logger.error("Must specify a model path.")
            logger.error("Must specify a CNN model path.")
        else:
            self.model_path = model_path

        self.img_num = 0

        if gpu is True:
            self.device = torch.device("cuda:{}".format(gpu_num))
            torch.jit.fuser('fuser2')
        else:
            self.device = torch.device("cpu")
            torch.jit.fuser('fuser1')

        self.classify = classify
        self.labels = labels

        self.n_imgs = n_imgs

        self.out_path = out_path

    def setup(self):
        ''' Initialize model
        '''
        os.makedirs(self.out_path, exist_ok=True)

        logger.info('Loading model for ' + self.name)
        self.done = False
        self.dropped_img = []

        t = time.time()        
        self.model = torch.jit.load(self.model_path).eval().to(self.device)
        torch.cuda.synchronize()
        load_model_time = (time.time() - t)*1000.0

        print('Time to load model:', load_model_time)
        with open(self.out_path + "load_model_time.txt", "w") as text_file:
            text_file.write("%s" % load_model_time)
            text_file.close()

        t = time.time()
        sample_input = torch.rand(size=(1, 3, 224, 224), device=self.device)
        with torch.no_grad():
            for _ in range(10):
                self.model(sample_input)
        
        torch.cuda.synchronize()
                
        warmup_time = (time.time() - t)*1000.0
        print('Time to warmup:', warmup_time)
        with open(self.out_path + "warmup_time.txt", "w") as text_file:
            text_file.write("%s" % warmup_time)
            text_file.close()

        if self.classify is True:
            with open(self.labels, "r") as f:
                self.labels = f.read().split(", ")
                f.close()

    def run(self):
        ''' Run the processor continually on input data, e.g.,images
        '''
        self.total_times = []

        if self.classify is True:
            self.true_label = []
            self.pred_label = []
            self.percent = []
            self.top_five = []

        with RunManager(self.name, self.runProcess, self.setup, self.q_sig, self.q_comm, runStore=self._getStoreInterface()) as rm:
            print(rm)
        
        print('Processor broke, avg time per image:', np.mean(self.total_times))
        print('Processor got through', self.img_num, ' images')

    def runProcess(self):
        ''' Run process. Runs once per image.
            Output is a location in the DS to continually
            place the processed image, model output, and classification/prediction with ref number that
            corresponds to the frame number (TODO)
            [From neurofinder/actors/processor.py]
        '''
        ids = self._checkInput()

        if ids is not None:
            t = time.time()
            self.done = False
            try:
                img = self.client.getID(ids[0])
                img = self._processImage(img)
                output, _, _ = self._runInference(img)
                features = output[0].to('cpu').numpy()
                predictions = output[1].to('cpu').numpy()
                if self.classify is True and self.labels is not None:
                    pred_label, percent, top_five = self._classifyImage(predictions, self.labels)
                    self.true_label.append(self.labels[self.client.getID(ids[1])])
                    self.pred_label.append(pred_label)
                    self.percent.append(percent)
                    self.top_five.append(top_five)

                self.img_num += 1
                self.total_times.append((time.time() - t)*1000.0)
                
            except ObjectNotFoundError:
                logger.error('Processor: Image {} unavailable from store, dropping'.format(self.img_num))
                self.dropped_img.append(self.img_num)
            except KeyError as e:
                logger.error('Processor: Key error... {0}'.format(e))
                self.dropped_img.append(self.img_num)
            except Exception as e:
                logger.error('Processor error: {}: {} during image number {}'.format(type(e).__name__,
                                                                                            e, self.img_num))
                print(traceback.format_exc())
                self.dropped_img.append(self.img_num)
            self.total_times.append((time.time() - t)*1000.0)
        else:
            pass

        if self.img_num == self.n_imgs:
            logger.error('Done processing all available data: {}'.format(self.img_num))
            self.data = None
            self.q_comm.put(None)
            self.done = True  # stay awake in case we get a shutdown signal

    def _checkInput(self):
        ''' Check to see if we have images for processing
        From basic demo
        '''
        try:
            res = self.q_in.get(timeout=0.005)
            return res
        #TODO: additional error handling
        except Empty:
            pass
            # logger.info('No images for processing')
            # return None

    def _processImage(self, img):
        ''' Load data - here, .jpg image to tensor
        Input is already loaded image from q_in
        '''
        if img is None:
            raise ObjectNotFoundError
        else:
            # Takes np img (HWC) -> (NHWC) -> (NCHW)
            img = torch.from_numpy(img.copy()).type(torch.FloatTensor)
            img = img.unsqueeze(dim=0).permute(0, 3, 1, 2)
        return img

    def _runInference(self, data):
        '''
        '''
        t = time.time()
        data = data.to(self.device)
        torch.cuda.synchronize()        
        to_device = time.time() - t
        with torch.no_grad():
            t = time.time()
            output = self.model(data)
            torch.cuda.synchronize()
            inf_time = time.time() - t
        return output, to_device, inf_time
        
    def _classifyImage(self, predictions, labels):
        '''
        '''
        index = np.argmax(predictions, axis=1)[0]  
        percentage = (softmax(predictions) * 100)[0]

        indices = np.argsort(predictions, axis=1)[::-1][0]
        top_five = [(labels[idx], percentage[idx].item()) for idx in indices[:5]]

        return labels[index], percentage[index], top_five

class NaNDataException(Exception):
    pass