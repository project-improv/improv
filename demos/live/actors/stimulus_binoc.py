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

class VisualStimulus(Actor):

    def __init__(self, *args, ip=None, port=None, seed=1234, stimuli = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.ip = ip
        self.port = port
        self.frame_num = 0
        self.displayed_stim_num = 0

        self.seed = 42 #1337 #81 #1337 #7419
        np.random.seed(self.seed)

        self.prepared_frame = None
        self.random_flag = False
        
        self.stimuli = np.load(stimuli, allow_pickle=True)
        np.save('output/generated_stimuli.npy', self.stimuli)

        self.initial = True
        # self.angle_set = [0, 45, 90, 135, 180, 225, 270, 315] #[0, 175, 100, 275, 50, 225, 125, 325] ##np.arange(0, 315, 45)
        # random.shuffle(self.angle_set)
        # self.which_angle = 0
        self.newN = False
        # self.vel_set = [0.02, 0.035, 0.05, 0.075, 0.10]
        # self.freq_set = [15, 30, 45]
        # random.shuffle(self.vel_set)
        # random.shuffle(self.freq_set)

        self.stim_choice = []
        self.GP_stimuli = []
        self.GP_stimuli_init = []
        self.stim_sets = []
        for i,s in enumerate(self.stimuli):
            print(i,s)
            self.stim_choice.append(s.shape[0])
    
            indices = np.arange(s.shape[0])
            random.shuffle(indices)        

            self.GP_stimuli_init.append(np.arange(s.shape[0])[indices])
            self.GP_stimuli.append(np.arange(s.shape[0]))
            # s2 = s.tolist()
            # random.shuffle(s2)
            self.stim_sets.append(s[indices].tolist())
        self.stim_choice = np.array(self.stim_choice)

        print('GP stimuli is ', self.GP_stimuli)

        snum = self.stim_choice[0]
        print('Number of angles is :', snum)
        self.grid_choice = np.arange(snum**2)
        np.random.shuffle(self.grid_choice)
        self.grid_choice = np.reshape(self.grid_choice, (snum,snum))
        print(self.grid_choice)
        self.grid_ind = np.arange(snum**2)

        self.initial_angles = np.linspace(0,360,endpoint=False, num=8)
        random.shuffle(self.initial_angles)
        self.which_angle = 0
        self.all_angles = self.stim_sets[0]
        random.shuffle(self.all_angles)

        ### Optimizer
        maxS = self.stim_choice #np.array([l[-1] for l in self.stim_choice])
        gamma = (1 / maxS) / 2
        print(gamma)
        var = 0.5 #1e-1
        nu = 1 #0.5 #1e-1
        eta = 5e-2 #8e-2 #-1e-2
        d = 2

        gp_copy = self.GP_stimuli.copy()
        xs = np.meshgrid(*gp_copy) #,x3,x4])
        x_star = np.empty(xs[0].shape + (d,))
        for i in range(d):
            x_star[...,i] = xs[i]

        self.x_star = x_star.reshape(-1, d)      #shape (a,d) where a is all possible test points
        print('Number of possible test points to optimize over: ', self.x_star.shape[0])

        self.optim = Optimizer(gamma[:d], var, nu, eta, self.x_star)
        init_T = 8
        self.X0 = np.zeros((d, init_T))
        # self.X0[0,:] = np.arange(8) #self.angle_set
        # self.X0[1,:] = 2 #self.stimuli[1][0]
        # self.X0[2,:] = 2 #32
        # X0 = self.stimuli[:,:init_T]
        # print('Stim sets', self.stim_sets)


        # self.X0[0,:] = self.GP_stimuli_init[0][:init_T]
        # self.X0[1,:] = self.GP_stimuli_init[1][:init_T]
        # self.X0[2,:] = self.GP_stimuli_init[2][:init_T]
        # print('Initial X0 will be ', self.X0)

        self.X = self.X0.copy()

        self.nID = None
        self.conf = None
        self.maxT = 20

        self.optimized_n = []

        self.saved_GP_est = []
        self.saved_GP_unc = []

        xs = np.meshgrid(*self.stimuli) #,x3,x4])
        x_star = np.empty(xs[0].shape + (d,))
        for i in range(d):
            x_star[...,i] = xs[i]

        self.stim_star = x_star.reshape(-1, d)

        print(self.stim_choice, self.GP_stimuli)

        self.stopping_list = []
        self.peak_list = []
        self.goback_neurons = []
        self.optim_f_list = []

        if self.random_flag:
            self.stimuli = self.stimuli.copy()[:, ::2]

            snum = int(self.stim_choice[0] / 2)
            print('Number of angles during grid phase since random is :', snum)
            self.grid_choice = np.arange(snum**2)
            np.random.shuffle(self.grid_choice)
            self.grid_choice = np.reshape(self.grid_choice, (snum,snum))
            print(self.grid_choice)
            self.grid_ind = np.arange(snum**2)


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

        np.save('output/optimized_neurons.npy', np.array(self.optimized_n))
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
                self.X = tmpX.copy()
                if tmpX.shape[1] > 8:
                    self.X = tmpX[:, -tmpX.shape[1]:]
                # print('self.X DIRECT from analysis is ', X, 'and self.X is ', self.X[:,-1])
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
            # # ## doing random stims
            if self.random_flag:
                
                if self.prepared_frame is None:
                    self.prepared_frame = self.random_frame()
                    # self.prepared_frame.pop('load')
                if (time.time() - self.timer) >= self.total_stim_time:
                        # self.random_frame()
                    self.send_frame(self.prepared_frame)
                    self.prepared_frame = None

            else:
                # print(self.optimized_n, set(self.optimized_n))
                nonopt = np.array(list(set(np.arange(self.y0.shape[0]))-set(self.optimized_n)))
                print('nonopt is ', nonopt)
                if len(nonopt) >= 1 or len(self.goback_neurons)>=1:
                    if len(nonopt) >= 1:
                        self.nID = nonopt[np.argmax(np.mean(self.y0[nonopt,:], axis=1))]
                        print('selecting most responsive neuron: ', self.nID)
                        self.optimized_n.append(self.nID)
                        self.saved_GP_est = []
                        self.saved_GP_unc = []
                    elif len(self.goback_neurons)>=1:
                        self.nID = self.goback_neurons.pop(0)
                        print('Trying again with neuron', self.nID)
                        self.optimized_n.append(self.nID)
                    
                    print(self.y0.shape, self.X.shape, self.X0.shape)
                    if self.X.shape[1] < self.y0.shape[1]:
                        self.optim.initialize_GP(self.X[:, :].T, self.y0[self.nID, -self.X.shape[1]:].T)
                    elif self.y0.shape[1] < self.maxT:
                        self.optim.initialize_GP(self.X[:, -self.y0.shape[1]:].T, self.y0[self.nID, -self.y0.shape[1]:].T)
                    else:
                        # self.optim.initialize_GP(self.X[:, -self.maxT:].T, self.y0[self.nID, -self.maxT:].T)
                        self.optim.initialize_GP(self.X[:, :].T, self.y0[self.nID, :].T)
                    print('known average sigma, ', np.mean(self.optim.sigma))
                    # self.optim.initialize_GP(self.X0[:, :3], self.y0[self.nID, :3])
                    self.test_count = 0
                    self.newN = False
                    self.stopping = np.zeros(self.maxT)

                    curr_unc = np.diagonal(self.optim.sigma.reshape((576,576))).reshape((24,24))
                    curr_est = self.optim.f.reshape((24,24))
                    self.saved_GP_unc.append(curr_unc)
                    self.saved_GP_est.append(curr_est)

                    ids = []
                    # print('--------------- nID', self.nID)
                    # ids.append(self.client.put(self.nID, 'nID'))
                    ids.append(self.nID)
                    ids.append(self.client.put(curr_est, 'est'))
                    ids.append(self.client.put(curr_unc, 'unc'))
                    # ids.append(self.client.put(self.conf, 'conf'))
                    self.q_out.put(ids)
                
                else:
                    self.initial = True

        ### update GP, suggest next stim
        else:
            
            if self.prepared_frame is None:
                X = np.zeros(2)
                # print('self.X from analysis is ', self.X[:,-1])
                # print('going back further', self.X)
                # print('GP_stimuli', self.GP_stimuli[0])
                # try:
                X[0] = self.GP_stimuli[0][int(self.X[0,-1])]
                X[1] = self.GP_stimuli[1][int(self.X[1,-1])]
                # X[2] = self.GP_stimuli[2][int(self.X[2,-1])]
                print('optim ', self.nID, ', update GP with', X, self.y0[self.nID, -1])
                self.optim.update_GP(np.squeeze(X), self.y0[self.nID,-1])

                curr_unc = np.diagonal(self.optim.sigma.reshape((576,576))).reshape((24,24))
                curr_est = self.optim.f.reshape((24,24))
                self.saved_GP_unc.append(curr_unc)
                self.saved_GP_est.append(curr_est)
                # except:
                #     pass
                ids = []
                # print('--------------- nID', self.nID)
                # ids.append(self.client.put(self.nID, 'nID'))
                ids.append(self.nID)
                ids.append(self.client.put(curr_est, 'est'))
                ids.append(self.client.put(curr_unc, 'unc'))
                # ids.append(self.client.put(self.conf, 'conf'))
                self.q_out.put(ids)

                stopCrit = self.optim.stopping()
                print('----------- stopCrit: ', stopCrit)
                self.stopping[self.test_count] = stopCrit
                self.test_count += 1

                
                if stopCrit < 3.0e-4: #6.0e-4: #8e-2: #0.37/2.05
                    peak = self.stim_star[np.argmax(self.optim.f)]
                    print('Satisfied with this neuron, moving to next. Est peak: ', peak)
                    # self.nID += 1
                    self.newN = True
                    self.stopping_list.append(self.stopping)
                    self.peak_list.append(peak)
                    self.optim_f_list.append(self.optim.f)

                    np.save('output/saved_GP_est_'+str(self.nID)+'.npy', np.array(self.saved_GP_est))
                    np.save('output/saved_GP_unc_'+str(self.nID)+'.npy', np.array(self.saved_GP_unc))

                    if len(self.optim_f_list) >= 50:
                        print('------------------ stopping optim, going random')
                        self.random_flag = True

                        # lower resolution
                        self.stimuli = self.stimuli.copy()[:, ::2]

                        snum = int(self.stim_choice[0] / 2)
                        print('Number of angles during grid phase is :', snum)
                        self.grid_choice = np.arange(snum**2)
                        np.random.shuffle(self.grid_choice)
                        self.grid_choice = np.reshape(self.grid_choice, (snum,snum))
                        print(self.grid_choice)
                        self.grid_ind = np.arange(snum**2)
                    
                elif self.test_count >= self.maxT:
                    self.goback_neurons.append(self.nID)
                    self.newN = True
                    self.stopping_list.append(self.stopping)
                    peak = self.stim_star[np.argmax(self.optim.f)]
                    self.peak_list.append(peak)
                    self.optim_f_list.append(self.optim.f)

                else:
                    ind, xt_1 = self.optim.max_acq()
                    print('suggest next stim: ', ind, xt_1, xt_1.T[...,None].shape)
                    # self.X = np.append(self.X, xt_1.T[...,None], axis=1)

                    self.prepared_frame = self.create_frame(ind)
                    # self.prepared_frame.pop('load')

            if (time.time() - self.timer) >= self.total_stim_time:
                self.send_frame(self.prepared_frame)
                self.prepared_frame = None


    def runAcquirer(self):
        ''' Main loop. If there're new files, read and put into store.
        '''
        #TODO: use poller instead to prevent blocking, include a timeout
        # try:
        #     ids = self.q_in.get(timeout=0.0001)
        #     X, Y, stim, _ = self.client.getList(ids)
        #     self.X = np.squeeze(np.array(X)).T
        #     # print(X)
        #     # try:
        #     #     Y = np.array(Y)
        #     #     print(Y.shape)
        #     # except:
        #     #     print('cannot convert y')
        #     try:
        #         b = np.zeros([len(Y),len(max(Y,key = lambda x: len(x)))])
        #         for i,j in enumerate(Y):
        #             b[i][:len(j)] = j
        #         # print(b.shape)
        #     except:
        #         pass
        #     # print(stim)
        #     # print(nID)

        #     self.y0 = b
        #     if not self.newN:
        #         self.optim.update_GP(self.X0[-1], self.y0[self.nID])

        #     if not self.initial and self.newN:
        #         self.nID = np.argmax(np.mean(b, axis=0))
        #         print('selecting most responsive neuron: ', self.nID)
        #         self.chooseFlag = True

        #     if self.nID and self.newN: # and nID != self.nID:
        #         # self.nID = nID
        #         self.new_neuron()
        #         self.newN = False

        #         ids = []
        #         # print('--------------- nID', self.nID)
        #         ids.append(self.client.put(self.nID, 'nID'))
        #         ids.append(self.client.put(self.conf, 'conf'))
        #         self.q_out.put(ids)
                

        #     # if stim is not None:
        #     #     self.send_frame(stim)

        # except Empty as e:
        #     pass
        # except Exception as e:
        #     print('error: {}'.format(e))

        # if self.chooseFlag:
        #     ind, xt_1 = self.optim.max_acq()
        #     print('suggest next stim: ', xt_1)

        # if not self.nID or self.chooseFlag:
        if self.prepared_frame is None:
            self.prepared_frame = self.random_frame()
            self.prepared_frame.pop('load')
        if (time.time() - self.timer) >= self.total_stim_time:
                # self.random_frame()
            self.send_frame(self.prepared_frame)
            self.prepared_frame = None


    # def new_neuron(self):
    #     # try:
    #     self.optim.initialize_GP(self.X0, self.y0[self.nID])
    #     # except:
    #     #     pass
    
    def send_frame(self, stim):
        if stim is not None:
            text = {'frequency':30, 'dark_value':0, 'light_value':250, 'texture_size':(1024,1024), 'texture_name':'grating_gray'}
            stimulus = {'stimulus': stim, 'texture': [text, text]}
            self._socket.send_string(self.stimulus_topic, zmq.SNDMORE)
            self._socket.send_pyobj(stimulus)
            self.timer = time.time()
            # print('Number of stimuli displayed: ', self.displayed_stim_num)
            self.displayed_stim_num += 1


    def create_frame(self, ind):
        xt = self.stim_star[ind]
        print('new stim frame ', ind, xt)
        angle = xt[0]
        angle2 = xt[1]
        vel = -0.01
        freq = 30
        light, dark = 250, 0

        stat_t = 10
        stim_t = stat_t + 5
        self.total_stim_time = stim_t
        center_width = 12
        center_x = 0
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
        snum = int(self.stim_choice[0] / 2)
        grid = np.argwhere(self.grid_choice==self.grid_ind[self.displayed_stim_num%(snum**2)])[0] #self.which_angle%24 #self.which_angle%24 #np.argwhere(self.grid_choice==self.grid_ind[self.displayed_stim_num%(36*36)])[0]
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

    period = 24

    for i in range(x.shape[0]):
        # K[:,i] = self.variance * rbf_kernel(x[:,i], x_j[:,i], gamma = self.gamma[i])
        for j in range(x_j.shape[0]):
            ## first dimension is direction
            # dist = np.abs(x[i,0] - x_j[j,0])
            # # print(dist)
            # # if dist > 12:
            # #     dist = 24 - dist
            # # print(dist)
            # K[i,j] = np.exp(-gamma[0]*((dist)**2))
            # K[i,j] *= variance * np.exp(-gamma[1:].dot((x[i,1:]-x_j[j,1:])**2))

            ## binocular
            dist1 = np.sin(np.pi * np.abs(x[i,0] - x_j[j,0]) / period)
            dist2 = np.sin(np.pi * np.abs(x[i,1] - x_j[j,1]) / period)

            # dist1 = np.abs(x[i,0] - x_j[j,0])
            # dist2 = np.abs(x[i,1] - x_j[j,1])

            K[i,j] = np.exp(-gamma[0]*(dist1**2))
            K[i,j] *= variance * np.exp(-gamma[1]*(dist2**2))

            
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

