import time
import os
import h5py
import random
import numpy as np
from skimage.io import imread

from improv.actor import Actor, RunManager

import logging; logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class Spike_Acquirer(Actor):

    def __init__(self, *args, filename=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.frame_num = 0
        self.data = None
        self.done = False
        self.flag = False
        self.N= 0
        self.filename = filename

    def setup(self):
        '''Get file names from config or user input
            Also get specified framerate, or default is 10 Hz
           Open file stream
           #TODO: implement more than h5 files
        '''        
        print('Looking for ', self.filename)
        if os.path.exists(self.filename):
            self.data = np.loadtxt(self.filename)
            print('Data length is ', self.data.shape[1])
        else: 
            raise FileNotFoundError

    def run(self):
        ''' Run indefinitely. Calls runAcquirer after checking for signals
        '''
        self.total_times = []
        self.timestamp = []

        with RunManager(self.name, self.runAcquirer, self.setup, self.q_sig, self.q_comm) as rm:
            print(rm)            
            
        print('Done running Acquire, avg time per frame: ', np.mean(self.total_times))
        print('Acquire got through ', self.frame_num, ' frames')
        np.savetxt('output/timing/acquire_frame_time.txt', np.array(self.total_times))
        np.savetxt('output/timing/acquire_timestamp.txt', np.array(self.timestamp))

    def runAcquirer(self):
        '''While frames exist in location specified during setup,
           grab frame, save, put in store
        '''

        if self.frame_num< self.data.shape[1]:
            before = self.frame_num-500 if self.frame_num > 500 else 0
            curr= self.data[:, :self.frame_num]

            S = self.data[np.any(curr, axis=1), before:self.frame_num] #.get_ordered()

            id = self.client.put(S, 'acq_spike'+str(self.frame_num))
            self.put([[id, 'acq_spike'+str(self.frame_num)]], save=[True])

            self.frame_num+=1

        else:
            pass



