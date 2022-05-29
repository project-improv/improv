import time
import os
import h5py
import random
import numpy as np
from skimage.io import imread

from improv.actor import Actor, RunManager

import scipy.io as scio

import logging; logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class RawBehavior(Actor):

    def __init__(self, *args, filename=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.frame_num = 0
        self.data = None
        self.done = False
        self.filename = filename

    def setup(self):
        '''Get file names from config or user input
            Also get specified framerate, or default is 10 Hz
           Open file stream
           #TODO: implement more than h5 files
        '''
        print('Looking for ', self.filename)
        if os.path.exists(self.filename):
            n, ext = os.path.splitext(self.filename)[:2]
            if ext == '.h5' or ext == '.hdf5':
                with h5py.File(self.filename, 'r') as file:
                    keys = list(file.keys())
                    self.data = file[keys[0]].value 
                    print('Behavior Data length is ', self.data.shape[2])

        else: 
            raise FileNotFoundError

    def run(self):

        with RunManager(self.name, self.runBehAcq, self.setup, self.q_sig, self.q_comm) as rm:
            print(rm)   

    def runBehAcq(self):

        if self.done:
            pass 
        elif(self.frame_num < self.data.shape[2]):
            frame = self.getFrame(self.frame_num)
            ## simulate frame-dropping
            # if self.frame_num > 1500 and self.frame_num < 1800:
            #     frame = None
            id = self.client.put(frame, 'beh'+str(self.frame_num))

            try:
                self.put([[id, str(self.frame_num)]], save=[True])
                self.frame_num += 1
                 #also log to disk #TODO: spawn separate process here?  
            except Exception as e:
                logger.error('Behavior Acquirer general exception: {}'.format(e))

            time.sleep(0.001) #pretend framerate

        else: # simulating a done signal from the source (eg, camera)
            logger.error('Done with all available frames: {0}'.format(self.frame_num))
            self.data = None
            self.q_comm.put(None)
            self.done = True

    def getFrame(self, num):
        ''' Here just return frame from loaded data
        '''

        return self.data[:,:,num]

class MotionBehavior(Actor):

    def __init__(self, *args, filename=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.frame_num = 0
        self.data = None
        self.done = False
        self.filename = filename

    def setup(self):
        '''Get file names from config or user input
            Also get specified framerate, or default is 10 Hz
           Open file stream
           #TODO: implement more than h5 files
        '''
        print('Looking for ', self.filename)
        if os.path.exists(self.filename):
            file= scio.loadmat(self.filename)
            self.data= np.squeeze(file['data'])
            print('Motion Data length is ', len(self.data))

        else: 
            raise FileNotFoundError

    def run(self):
    
        with RunManager(self.name, self.runBehAcq, self.setup, self.q_sig, self.q_comm) as rm:
            print(rm)  

    def runBehAcq(self):
    
        if self.done:
            pass 
        elif(self.frame_num < len(self.data)):
            frame = self.getFrame(self.frame_num)
            ## simulate frame-dropping
            # if self.frame_num > 1500 and self.frame_num < 1800:
            #     frame = None
            id = self.client.put(frame, 'beh'+str(self.frame_num))

            try:
                self.put([[id, str(self.frame_num)]], save=[True])
                self.frame_num += 1
                 #also log to disk #TODO: spawn separate process here?  
            except Exception as e:
                logger.error('Motion Acquirer general exception: {}'.format(e))

            time.sleep(0.001) #pretend framerate

        else: # simulating a done signal from the source (eg, camera)
            logger.error('Done with all available frames: {0}'.format(self.frame_num))
            self.data = None
            self.q_comm.put(None)
            self.done = True

    def getFrame(self, num):
        ''' Here just return frame from loaded data
        '''

        return self.data[num]