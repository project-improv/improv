import os
import zmq
import time
import h5py
import random
import numpy as np
from skimage.io import imread

from improv.actor import Actor, RunManager, Spike

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
                    print('Data length of ', self.filename,' is ', len(self.data))


        else: raise FileNotFoundError

        print('Loading data/raw_C.txt')
        x = np.loadtxt('data/raw_C.txt')
        self.raw_C = np.transpose(x) # transposed so each frame is an element
        print('Data length of data/raw_C.txt is ', len(self.raw_C))

        self.setupZMQ()

    def run(self):
        ''' Run indefinitely. Calls runAcquirer after checking for signals
        '''
        self.total_times = []
        self.timestamp = []

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
        elif((self.frame_num < len(self.data)) and (self.frame_num < len(self.raw_C))):
            frame = self.getFrame(self.frame_num)
            ## simulate frame-dropping
            # if self.frame_num > 1500 and self.frame_num < 1800:
            #     frame = None
            t= time.time()
            
            # data from given file (currently it's the image data)
            id = self.client.put(frame, 'acq_raw'+str(self.frame_num))
            t1= time.time()
            self.timestamp.append([time.time(), self.frame_num])

            # raw_C data
            timepiece = self.raw_C[self.frame_num]
            id2 = self.client.put(timepiece, 'acq_raw_timepiece'+str(self.frame_num))
            
            print('Put into store:', id)

            # TODO: encode ids to successfully send at the same time through zmq (maybe use pyarrow encoding/decoding)
            zmqlist = [id.binary(),id2.binary()] # or create a dict
            # print(zmqlist) # checking format

            try:
                self.socket.send(id.binary()) # id.binary() for image data and id2.binary() for raw_C data
                # self.socket.send(zmqlist) # TODO: still working on sending both id's at once
                print("Message sent to jupyter notebook!")
                time.sleep(.3)
                logger.info(self.frame_num)
                self.frame_num += 1
           
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

            self.socket.send_string(Spike.quit())

    def getFrame(self, num):
        ''' Here just return frame from loaded data
        '''
        return self.data[num,:,:]

    def saveFrame(self, frame):
        ''' Save each frame via h5 dset
        '''
        self.dset[self.frame_num-1] = frame
        self.f.flush()

if __name__=="__main__":
    #starting the store manually
    import subprocess
    from improv.store import Limbo
    p = subprocess.Popen(["plasma_store", "-s", "/tmp/store", "-m", str(10000000000)],\
    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    # 1000000000 for id and 10000000000 for id2

    testfile = FileAcquirerZMQ(filename='data/tbif_ex_crop.h5', framerate=5, name='j')
    # TODO: add the second file (currently the raw_C data) as another attribute to FileAcquirerZMQ
    
    lmb = Limbo(store_loc = "/tmp/store")
    testfile.setStore(lmb)
    
    print("The store has started")
    testfile.setup()
    testfile.run()
    print("run() is finished")

    p.kill()




