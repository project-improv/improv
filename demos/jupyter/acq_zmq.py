import time
import os
import h5py
import random
import numpy as np
from skimage.io import imread

import zmq

from improv.actor import Actor, RunManager

import logging; logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Classes: File Acquirer for ZMQ

class FileAcquirerZMQ(Actor):
    '''Class to import data from files and output
       frames in a buffer, or discrete.
    '''
    def __init__(self, *args, filename=None, framerate=30, **kwargs):
        super().__init__(*args, **kwargs)
        self.frame_num = 0
        self.data = None
        self.done = False
        self.flag = False
        self.filename = filename
        self.framerate = 1/framerate

    def setupZMQ(self):
        ''' Setting up a port and socket for zmq messaging
        '''
        
        context = zmq.Context()
        self.socket = context.socket(zmq.PUB)
        self.socket.bind("tcp://127.0.0.1:5555")
        # message = self.socket.recv()
        # print("Received request from jupyter: ", message)
    
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
                    self.data = file[keys[0]][()]
                    print('Data length is ', len(self.data))

        else: raise FileNotFoundError

        x = np.loadtxt('data/raw_C.txt')
        self.raw_C = np.transpose(x)

        self.setupZMQ()

    def run(self):
        ''' Run indefinitely. Calls runAcquirer after checking for signals
        '''
        self.total_times = []
        self.timestamp = []

        # with RunManager(self.name, self.runAcquirer, self.setup, self.q_sig, self.q_comm) as rm:
        #     print(rm)
        while True:
            self.runAcquirer()


        print('Done running Acquire, avg time per frame: ', np.mean(self.total_times))
        print('Acquire got through ', self.frame_num, ' frames')
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
        np.savetxt('output/timing/acquire_frame_time.txt', np.array(self.total_times))
        np.savetxt('output/timing/acquire_timestamp.txt', np.array(self.timestamp))

    def runAcquirer(self):
        '''While frames exist in location specified during setup,
           grab frame, save, put in store
        '''
        t = time.time()

        if self.done:
            pass
        elif(self.frame_num < len(self.data)):
            frame = self.getFrame(self.frame_num)
            ## simulate frame-dropping
            # if self.frame_num > 1500 and self.frame_num < 1800:
            #     frame = None
            t= time.time()
            # print("runAcq is going") # delete this
            id = self.client.put(frame, 'acq_raw'+str(self.frame_num))
            print("id has been put into the store") #delete this
            t1= time.time()
            self.timestamp.append([time.time(), self.frame_num])

            timepiece = self.raw_C[self.frame_num]
            id2 = self.client.put(timepiece, 'acq_raw_timepiece'+str(self.frame_num))
            print(id2)
            zmqlist = id.binary() #[id.binary(),id2.binary()]
            print(zmqlist)

            try:
                # self.put([[id, str(self.frame_num)]], save=[True])
                # print("before sending thru zmq") # delete this
                print(id)
                # self.socket.send(id.binary())
                self.socket.send(zmqlist)
                time.sleep(.2)
                # self.socket.send_string("message sent from server!")
                print("socket sent message out")
                logger.info(self.frame_num)
                self.frame_num += 1
                    #also log to disk #TODO: spawn separate process here?
           
            except Exception as e:
                logger.error('Acquirer general exception: {}'.format(e))

            time.sleep(self.framerate) #pretend framerate
            self.total_times.append(time.time()-t)

        else: # simulating a done signal from the source (eg, camera)
            logger.error('Done with all available frames: {0}'.format(self.frame_num))
            self.data = None
            # self.q_comm.put(None)
            self.done = True # stay awake in case we get e.g. a shutdown signal
            #if self.saving:
            #    self.f.close()

    def getFrame(self, num):
        ''' Here just return frame from loaded data
        '''
        return self.data[num,:,:]

    def saveFrame(self, frame):
        ''' Save each frame via h5 dset
        '''
        self.dset[self.frame_num-1] = frame
        self.f.flush()

# class StimAcquirer(Actor):
#     ''' Class to load visual stimuli data from file
#         and stream into the pipeline
#     '''
#     def __init__(self, *args, param_file=None, filename=None, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.param_file = param_file
#         self.filename= filename

#     def setup(self):
#         self.n= 0
#         self.sID = 0
#         if os.path.exists(self.filename):
#             print('Looking for ', self.filename)
#             n, ext = os.path.splitext(self.filename)[:2]
#             if ext== ".txt":
#                 # self.stim = np.loadtxt(self.filename)
#                 self.stim=[]
#                 f = np.loadtxt(self.filename)
#                 for _, frame in enumerate(f):
#                     stiminfo = frame[0:2]
#                     self.stim.append(stiminfo)
#             else:
#                 logger.error('Cannot load file, possible bad extension')
#                 raise Exception

#         else: raise FileNotFoundError

#     def run(self):
#         ''' Run continuously, waiting for input
#         '''
#         with RunManager(self.name, self.getInput, self.setup, self.q_sig, self.q_comm) as rm:
#             logger.info(rm)

#     def getInput(self):
#         ''' Check for input from behavioral control
#         '''
#         if self.n<len(self.stim):
#             # s = self.stim[self.sID]
#             # self.sID+=1
#             self.q_out.put({self.n:self.stim[self.n]})
#         time.sleep(0.5)   # simulate a particular stimulus rate
#         self.n+=1

if __name__=="__main__":
    #start the store manually (dont start in jupyter)
    import subprocess
    from improv.store import Limbo
    p = subprocess.Popen(["plasma_store", "-s", "/tmp/store", "-m", str(1000000000)],\
    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    testfile = FileAcquirerZMQ(filename='data/tbif_ex_crop.h5', framerate=5, name='j')
    lmb = Limbo(store_loc = "/tmp/store")
    testfile.setStore(lmb)
    
    print("the store has started")
    testfile.setup()
    testfile.run()
    print("run() is finished")

    p.kill()

    #put in store
    #send id

    #in jupyter
    #get id

