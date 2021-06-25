import time
import os
import h5py
import struct
import numpy as np
import random
import ipaddress
import zmq
import json
from pathlib import Path
from improv.actor import Actor, Spike, RunManager
from queue import Empty

import logging; logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class VisualStimulus(Actor):

    def __init__(self, *args, ip=None, port=None, seed=42, **kwargs):
        super().__init__(*args, **kwargs)
        self.ip = ip
        self.port = port
        self.frame_num = 0

        self.seed = seed
        np.random.seed(self.seed)

        # Sanity check
        # ipaddress.ip_address(self.ip)  # Check if IP is valid.
        # if not 0 <= port <= 65535:
        #     raise ValueError(f'Port {self.port} invalid.')

        # self.context = zmq.Context()
        # self.socket = self.context.socket(zmq.SUB)
        # self.socket.connect(f"tcp://{self.ip}:{self.port}")
        # self.socket.setsockopt_string(zmq.SUBSCRIBE, self.topicfilter)

    def setup(self):
        context = zmq.Context()
        
        self._socket = context.socket(zmq.PUB)
        send_IP =  self.ip
        send_port = self.port
        self._socket.bind('tcp://' + str(send_IP)+":"+str(send_port))
        self.stimulus_topic = 'stim'

        self.timer = time.time()

    def run(self):
        '''Triggered at Run
        '''
        self.total_times = []
        self.timestamp = []
        self.stimmed = []
        self.frametimes = []
        self.framesendtimes = []
        self.stimsendtimes = []
        self.tailsendtimes = []
        self.tails = []

        with RunManager(self.name, self.runAcquirer, self.setup, self.q_sig, self.q_comm) as rm:
            print(rm)
        print('-------------------------------------------- stim')


        print('Stimulus complete, avg time per frame: ', np.mean(self.total_times))
        print('Stim got through ', self.frame_num, ' frames')
        

    def runAcquirer(self):
        ''' Main loop. If there're new files, read and put into store.
        '''
        #TODO: use poller instead to prevent blocking, include a timeout
        try:
            stim = self.q_in.get(timeout=0.0001)

            if stim is not None:
                self.send_frame(stim)

        except Empty as e:
            pass
        except Exception as e:
            print('error: {}'.format(e))

        if (time.time() - self.timer) >= 30:
                self.random_frame()

    
    def send_frame(self, stim):
        self._socket.send_string(self.stimulus_topic, zmq.SNDMORE)
        self._socket.send_pyobj(stim)

        self.timer = time.time()


    def random_frame(self):
        angle = np.random.choice(np.arange(0, 350, 10)) #[0, 45, 90, 135, 180, 225, 270, 315])
        vel = -np.around(np.random.choice(np.linspace(0.02, 0.2, num=18)), decimals=2)
            #np.random.choice([0, -0.02, -0.04, -0.08, -0.1])/10 #, -0.12, -0.14, -0.16, -0.18, -0.2])
        stat_t = 15
        stim_t = 10
        freq = np.random.choice(np.arange(5,80,5))
        light, dark = self.contrast(np.random.choice(5))
        center_width = 16
        center_x = 0
        center_y = 0  
        strip_angle = 0

        stim = {
                'stim_type': 's', 'angle': angle, #angle],
                'velocity': vel, #vel],
                'stationary_time': stat_t, #stat_t],
                'stim_time': stim_t, #stim_t],
                'frequency': freq, #freq],
                'lightValue': light,
                'darkValue': dark,
                'center_width' : center_width,
                'center_x': center_x,
                'center_y': center_y,
                'strip_angle': strip_angle
                    }

        self._socket.send_string(self.stimulus_topic, zmq.SNDMORE)
        self._socket.send_pyobj(stim)

        print('Sent random stimulus: ', stim)

        self.timer = time.time()

    def contrast(self, n):
        if n==0:
            light = 0
            dark = 30
        elif n==1:
            light = 0
            dark = 120
        elif n==2:
            light = 0
            dark = 240
        elif n==3:
            light = 120
            dark = 240
        elif n==4:
            light = 210
            dark = 240
        else:
            print('No pre-selected contrast found; using default (3)')
            light = 0
            dark = 240

        return light, dark