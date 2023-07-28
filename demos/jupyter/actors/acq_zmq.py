import os
import zmq
import time
import h5py
import random
import numpy as np
from skimage.io import imread
from improv.store import StoreInterface

from improv.actor import Actor, RunManager, Signal

import logging; logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


max_frames = 300

class Processor(Actor):
    '''Class to import data from files and output
       frames in a buffer, or discrete.
    '''
    def __init__(self, *args, filename=None, framerate=30, **kwargs):
        super().__init__(*args, **kwargs)

        self.name = "Processor"
        self.frame = None
        self.frame_num = 1

    def setupZMQ(self):
        ''' Setting up a port and socket for zmq messaging
        '''
        
        context = zmq.Context()
        self.socket = context.socket(zmq.PUB)
        self.socket.bind("tcp://127.0.0.1:5556")
    
    def setup(self):
        '''Setup ZMQ to send data to jupyter notebook
        '''
        
        logger.info("Completed setup for Processor")
        self.setupZMQ()

    def runStep(self):
        '''While frames exist in location specified during setup,
           grab frame, save, put in store
        '''

        frame = None
        try:
            frame = self.q_in.get(timeout=0.001)

        except:
            logger.error("Could not get frame!")
            pass

        if frame is not None and self.frame_num is not None:
            self.frame = self.client.getID(frame[0][0])[0]
            id = self.client.put(self.frame, 'acq_raw'+str(self.frame_num))
           
            try:
                self.socket.send(id.binary()) # id.binary() for image data and id2.binary() for raw_C data
                # self.socket.send(zmqlist) # TODO: still working on sending both id's at once
                logger.info("Message sent to jupyter notebook!")
                time.sleep(.3)
                logger.info(self.frame_num)
                self.frame_num += 1
           
            except Exception as e:
                logger.error('Acquirer general exception: {}'.format(e))


        if self.frame_num == max_frames:
             # simulating a done signal from the source (eg, camera)
            logger.info('Done with all available frames: {0}'.format(self.frame_num))
            self.data = None
            # self.q_comm.put(None)
            self.done = True # stay awake in case we get e.g. a shutdown signal
            #if self.saving:
            #    self.f.close()

            self.socket.send_string(Signal.quit())

    def getFrame(self, num):
        ''' Here just return frame from loaded data
        '''
        return self.data[num,:,:]

    def saveFrame(self, frame):
        ''' Save each frame via h5 dset
        '''
        self.dset[self.frame_num-1] = frame
        self.f.flush()
