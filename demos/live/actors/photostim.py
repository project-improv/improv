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
from scipy.stats import norm
import random

import logging; logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class PhotoStimulus(Actor):

    def __init__(self, *args, ip=None, port=None, seed=1234, stimuli = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.ip = ip
        self.port = port
        self.frame_num = 0
        self.displayed_stim_num = 0

        self.seed = 1337 #81 #1337 #7419
        np.random.seed(self.seed)

        self.prepared_frame = None

        self.stopping_list = []
        self.peak_list = []
        self.goback_neurons = []
        self.optim_f_list = []

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

        with RunManager(self.name, self.runStimulusSelector, self.setup, self.q_sig, self.q_comm) as rm:
            print(rm)
        print('-------------------------------------------- stim')

        # np.save('output/optimized_neurons.npy', np.array(self.optimized_n))
        # print(self.stopping_list)
        np.save('output/stopping_list.npy', np.array(self.stopping_list))
        # print(self.peak_list)
        np.save('output/peak_list.npy', np.array(self.peak_list))
        # print(self.optim_f_list)
        np.save('output/optim_f_list.npy', np.array(self.optim_f_list))

        print('Stimulus complete, avg time per frame: ', np.mean(self.total_times))
        print('Stim got through ', self.frame_num, ' frames')
        
    def runStimulusSelector(self):
        ### Get data from analysis actor
        try:
            ids = self.q_in.get(timeout=0.0001)
            X, Y, stim, _ = self.client.getList(ids)
            tmpX = np.squeeze(np.array(X)).T
            # print(tmpX.shape, '----------------------------------------------------')
            sh = len(tmpX.shape)
            if sh > 1:
                if tmpX.shape[1] > 8:
                    self.X = self.X[:, -tmpX.shape[1]:]
                    # print('truncating')
            # print(self.X)

            try:
                b = np.zeros([len(Y),len(max(Y,key = lambda x: len(x)))])
                for i,j in enumerate(Y):
                    b[i][:len(j)] = j
                self.y0 = b.T
                # print('X, Y shapes: ', self.X.shape, self.y0.shape)
            except:
                pass
            

        except Empty as e:
            pass
        except Exception as e:
            print('Error in stimulus get: {}'.format(e))


        ### initial run ignores signals and just sends 8 basic stimuli
        if self.initial:
            if self.prepared_frame is None:
                self.prepared_frame = self.initial_frame()
                # self.prepared_frame.pop('load')
            if (time.time() - self.timer) >= self.total_stim_time:
                self.send_frame(self.prepared_frame)
                self.prepared_frame = None


        ### once initial done, or we move on, initial GP with next neuron
        elif self.newN:
            ## doing random stims
            if self.prepared_frame is None:
                self.prepared_frame = self.random_frame()
                # self.prepared_frame.pop('load')
            if (time.time() - self.timer) >= self.total_stim_time:
                    # self.random_frame()
                self.send_frame(self.prepared_frame)
                self.prepared_frame = None

    
    def send_frame(self, stim):
        # text = {'frequency':30, 'dark_value':0, 'light_value':250, 'texture_size':(1024,1024), 'texture_name':'grating_gray'}
        # stimulus = {'stimulus': stim, 'texture': [text, text]}
        stimulus = self.coords[self.current_neuron]
        self._socket.send_string(self.stimulus_topic, zmq.SNDMORE)
        self._socket.send_pyobj(stimulus)
        self.timer = time.time()
        print('Number of neurons targeted: ', self.displayed_stim_num)
        self.displayed_stim_num += 1


    def create_frame(self, ind):
        xt = self.stim_star[ind]
        print('new frame ', ind, xt)
        angle = xt[0]
        vel = -xt[1]
        freq = xt[2]
        light, dark = 240, 0

        stat_t = 10
        stim_t = stat_t + 2
        self.total_stim_time = stim_t
        center_width = 16
        center_x = 0.1
        center_y = 0  
        strip_angle = 0

        stim = {
                'load': 0,
                'stim_type': 's', 'angle': angle, #angle],
                'velocity': vel, #vel],
                'stationary_time': stat_t, #stat_t],
                'duration': stim_t, #stim_t],
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

        self.timer = time.time()
        return stim

    def initial_frame(self):
        if self.which_angle%8 == 0:
            random.shuffle(self.initial_angles)
        angle = self.initial_angles[self.which_angle%8] #self.stim_sets[0][self.which_angle%len(self.stim_sets[0])]
        vel = -0.01 #self.stim_sets[1][4]#[self.which_angle%len(self.stim_sets[1])] #self.stimuli[1][2]
        freq = 30 #self.stim_sets[2][2] #[self.which_angle%len(self.stim_sets[2])] #30

        ## random sampling for initialization
        initial_length = 16
        # angle = np.random.choice(self.stimuli[0])
        # vel = -np.random.choice(self.stimuli[1])
        # freq = np.random.choice(self.stimuli[2])
        light, dark = 250, 0

        self.which_angle += 1
        if self.which_angle >= initial_length: 
            self.initial = False
            self.newN = True
            self.which_angle = 0
            logger.info('Done with initial frames, starting random set')
        
        stat_t = 10
        stim_t = stat_t + 5
        self.total_stim_time = stim_t
        center_width = 12
        center_x = 0#.1
        center_y = 0.015 
        strip_angle = 0

        stim = {
                'stim_name': 'stim_name',
                'angle': (angle, angle),
                'velocity': (vel, vel),
                'stationary_time': (stat_t, stat_t),
                'duration': (stim_t, stim_t),
                'frequency': (freq, freq),
                'light_value': (light, light),
                'dark_value': (dark, dark),
                'strip_width' : center_width,
                'position': (center_x, center_y),
                'strip_angle': strip_angle
                    }

        # self._socket.send_string(self.stimulus_topic, zmq.SNDMORE)
        # self._socket.send_pyobj(stim)

        self.timer = time.time()
        return stim


    def random_frame(self):

        # if self.initial:
        #     angle = self.angle_set[self.which_angle%8]
        #     vel = -self.stimuli[1][0]
        #     freq = 30
        #     light, dark = 240, 0

        #     self.which_angle += 1
        #     if self.which_angle >= (8*2): 
        #         self.initial = False
        #         self.newN = True
        
        # if self.which_angle%24==0:
        #     random.shuffle(self.all_angles)

        ## grid choice
        grid = np.argwhere(self.grid_choice==self.grid_ind[self.displayed_stim_num%(24*24)])[0] #self.which_angle%24 #np.argwhere(self.grid_choice==self.grid_ind[self.displayed_stim_num%(36*36)])[0]
        angle = self.stimuli[0][grid[0]] #self.all_angles[grid] #self.stimuli[0][grid[0]]
        angle2 = self.stimuli[0][grid[1]]

        # else:
            ## stimuli has angle, vel, freq, contrast in that order
        # angle = np.random.choice(self.stimuli[0])
        # angle2 = np.random.choice(self.stimuli[0])
        vel = -0.01 #np.random.choice(self.stimuli[1])
        freq = 30 #np.random.choice(self.stimuli[2])
        light, dark = 250, 0 # self.contrast(np.random.choice(self.stimuli[3]))

        # angle = self.angle_set[self.which_angle%16] #np.random.choice() #[0, 45, 90, 135, 180, 225, 270, 315])
        # self.which_angle += 1
        # angle = int(np.random.choice(np.linspace(0,350,num=15)))
        # vel = -0.02 #np.around(np.random.choice(np.linspace(0.02, 0.1, num=4)), decimals=2)
        # vel = -0.02 #np.random.choice(np.array([-0.02, -0.06, -0.1])) #, -0.12, -0.14, -0.16, -0.18, -0.2])
        stat_t = 10
        stim_t = stat_t + 5
        self.total_stim_time = stim_t
        # freq = 60 #np.random.choice(np.array([20,40,60])) #np.arange(5,80,5))
        # light, dark = 0, 240 #self.contrast(np.random.choice(5))
        center_width = 12
        center_x = 0#0.01
        center_y = 0.015  
        strip_angle = 0

        stim = {
                'stim_name': 'stim_name',
                'angle': (angle, angle2),
                'velocity': (vel, vel),
                'stationary_time': (stat_t, stat_t),
                'duration': (stim_t, stim_t),
                'frequency': (freq, freq),
                'light_value': (light, light),
                'dark_value': (dark, dark),
                'strip_width' : center_width,
                'position': (center_x, center_y),
                'strip_angle': strip_angle
                    }

        # self._socket.send_string(self.stimulus_topic, zmq.SNDMORE)
        # self._socket.send_pyobj(stim)

        # print('Sent stimulus to be loaded: ', stim)

        self.timer = time.time()
        return stim

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

        return dark, light


class Optimizer():
    def __init__(self, gamma, var, nu, eta, x_star):
        self.gamma = gamma
        self.variance = var
        self.nu = nu
        self.eta = eta
        self.x_star = x_star        

        self.d = self.x_star.shape[1]

        self.f = None
        self.sigma = None       ## Note: this is actually sigma squared
        self.X_t = None
        self.K_t = None
        self.k_star = None
        self.y = None
        self.A = None

        self.t = 0

    # def kernel(self, x, x_j):
    #     ## x shape: (T, d) (# tests, # dimensions)
    #     K = np.zeros((x.shape[0], x_j.shape[0]))
    #     for i in range(x.shape[0]):
    #         # K[:,i] = self.variance * rbf_kernel(x[:,i], x_j[:,i], gamma = self.gamma[i])
    #         for j in range(x_j.shape[0]):
    #             K[i,j] = self.variance * np.exp(-self.gamma.dot((x[i,:]-x_j[j,:])**2))
    #     return K


    def initialize_GP(self, X, y):
        ## X is a matrix (T,d) of initial T measurements we have results for

        # self.mu = 0
        # self.sigma = self.kernel(x, x_j)

        self.X_t = X
        self.y = y
        # print('X_t, y, x_star shapes: ', self.X_t.shape, self.y.shape, self.x_star.shape)

        T = self.X_t.shape[0]
        a = self.x_star.shape[0]

        self.test_count = np.zeros(a)

        self.K_t = kernel(self.X_t, self.X_t, self.variance, self.gamma)
        self.k_star = kernel(self.X_t, self.x_star, self.variance, self.gamma)

        self.A = np.linalg.inv(self.K_t + self.eta**2 * np.eye(T))
        self.f = self.k_star.T @ self.A @ self.y
        self.sigma = self.variance * np.eye(a) - self.k_star.T @ self.A @ self.k_star
        ### TODO: rewrite sigma computation to be every a not matrix mult
        # self.sigma = np.diagonal(self.sigma)

        self.t = T

    def update_obs(self, x, y):
        self.y_t1 = np.array([y])
        self.x_t1 = x[None,...]
       
    def update_GP(self, x, y):
        self.update_obs(x, y)

        ## Can't do internally due to out of memory / invalid array errors from numpy
        self.k_t, self.u, self.phi, f_upd, sigma_upd = update_GP_ext(self.X_t, self.x_t1, self.A, self.x_star, self.eta, self.y, self.y_t1, self.k_star, self.variance, self.gamma)

        # print('Mean f upd: ', np.mean(f_upd), np.mean(self.f))
        # print('Mean sigma upd: ', np.mean(np.diagonal(sigma_upd)), np.mean(np.diagonal(self.sigma)))

        self.f = self.f + f_upd
        # self.sigma = self.sigma + np.diagonal(sigma_upd)
        # self.f = self.k_star.T @ self.A @ self.y
        sigma = self.variance * np.eye(self.x_star.shape[0]) - self.k_star.T @ self.A @ self.k_star
        # self.sigma = np.diagonal(sigma)

        self.iterate_vars()

    def iterate_vars(self):
        self.y = np.append(self.y, self.y_t1)
        self.X_t = np.append(self.X_t, self.x_t1, axis=0)
        self.k_star = np.append(self.k_star, kernel(self.x_t1, self.x_star, self.variance, self.gamma), axis=0)

        ## update for A
        self.A = self.A + self.phi * np.outer(self.u, self.u)
        self.A = np.vstack((self.A, -self.phi*self.u.T))
        right = np.append(-self.phi*self.u, self.phi)
        self.A = np.column_stack((self.A, right))

        self.t += 1

    def max_acq(self):
        test_pt = np.argmax(self.ucb())
        
        if self.test_count[test_pt] > 5:
            test_pt = np.random.choice(np.arange(self.x_star.shape[0]))
            print('choosing random stim instead')
        self.test_count[test_pt] += 1

        return test_pt, self.x_star[test_pt]

    def ucb(self):
        tau = self.d * np.log(self.t + 1e-16)
        # import pdb; pdb.set_trace()
        sig = self.sigma
        if np.any(sig < 0):
            sig = np.clip(sig, 0, np.max(sig))
        fcn = self.f + np.sqrt(self.nu * tau) * np.sqrt(np.diagonal(sig))
        return fcn

    def stopping(self):
        val = self.f - np.max(self.f) - 1e-4
        # PI = np.max(norm.cdf((val) / (np.diagonal(self.sigma))))

        # using expected improvement
        sig = np.diagonal(self.sigma)
        EI = np.max(val * norm.cdf(val / sig) + sig * norm.pdf(val))
        return EI


def kernel(x, x_j, variance, gamma):
    ## x shape: (T, d) (# tests, # dimensions)
    K = np.zeros((x.shape[0], x_j.shape[0]))
    # print('dimensions internal ', x_j.shape[1])
    # print('min max ', np.min(x_j[:,0]), np.max(x_j[:,0]))
    # print('min max ', np.min(x_j[:,1]), np.max(x_j[:,1]))
    # print('min max ', np.min(x_j[:,2]), np.max(x_j[:,2]))

    # print('dimensions internal ', x.shape[1])
    # print('min max ', np.min(x[:,0]), np.max(x[:,0]))
    # print('min max ', np.min(x[:,1]), np.max(x[:,1]))
    # print('min max ', np.min(x[:,2]), np.max(x[:,2]))

    for i in range(x.shape[0]):
        # K[:,i] = self.variance * rbf_kernel(x[:,i], x_j[:,i], gamma = self.gamma[i])
        for j in range(x_j.shape[0]):
            ## first dimension is direction
            dist = np.abs(x[i,0] - x_j[j,0])
            # print(dist)
            if dist > 8:
                dist = 16 - dist
            # print(dist)
            K[i,j] = np.exp(-gamma[0]*((dist)**2))
            K[i,j] *= variance * np.exp(-gamma[1:].dot((x[i,1:]-x_j[j,1:])**2))
    return K

def update_GP_ext(X_t, x_t1, A, x_star, eta, y, y_t1, k_star, variance, gamma):

    k_t = kernel(X_t, x_t1, variance, gamma)
    u = A @ k_t
    k_t1 = kernel(x_t1, x_t1, variance, gamma)
    k_star_t1 = kernel(x_t1, x_star, variance, gamma)
    phi = np.linalg.inv(k_t1 + eta**2 - k_t.T.dot(u))
    kuk = k_star.T @ u - k_star_t1.T
    f = np.squeeze(phi * kuk * (y.dot(u) - y_t1))
    sigma = phi * (kuk**2)
    # import pdb; pdb.set_trace()

    return k_t, u, phi, f, sigma 

