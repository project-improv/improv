import time
import os
import h5py
import struct
import numpy as np
import random
from pathlib import Path
from skimage.io import imread
from improv.actor import Actor, Signal, RunManager
from queue import Empty

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class FileAcquirer(Actor):
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

    def setup(self):
        '''Get file names from config or user input
            Also get specified framerate, or default is 10 Hz
           Open file stream
           #TODO: implement more than h5 files
        ''' 
        logger.info("Running setup for " + self.name)       
        print('Looking for ', self.filename)
        if os.path.exists(self.filename):
            n, ext = os.path.splitext(self.filename)[:2]
            if ext == '.h5' or ext == '.hdf5':
                with h5py.File(self.filename, 'r') as file:
                    keys = list(file.keys())
                    self.data = file[keys[0]].value 
                    print('Data length is ', len(self.data))

        else: raise FileNotFoundError

        #if self.saving:
        #    save_file = self.filename.split('.')[0]+'_backup'+'.h5'
        #    self.f = h5py.File(save_file, 'w', libver='latest')
        #    self.dset = self.f.create_dataset("default", (len(self.data),)) #TODO: need to set maxsize to none?

    def run(self):
        ''' Run indefinitely. Calls runAcquirer after checking for signals
        '''
        self.total_times = []
        self.timestamp = []
        #self.links = {}
        self.links['q_sig'] = self.q_sig
        self.links['q_comm'] = self.q_comm
        self.actions = {}
        self.actions['run'] = self.runAcquirer
        self.actions['setup'] = self.setup
        with RunManager(self.name, self.actions, self.links) as rm:
            print(rm)            
            
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
            id = self.client.put(frame, 'acq_raw'+str(self.frame_num))
            t1= time.time()
            self.timestamp.append([time.time(), self.frame_num])
            try:
                self.put([[id, str(self.frame_num)]], save=[True])
                self.frame_num += 1
                 #also log to disk #TODO: spawn separate process here?  
            except Exception as e:
                logger.error('Acquirer general exception: {}'.format(e))

            time.sleep(self.framerate) #pretend framerate
            self.total_times.append(time.time()-t)

        else: # simulating a done signal from the source (eg, camera)
            logger.error('Done with all available frames: {0}'.format(self.frame_num))
            self.data = None
            self.q_comm.put(None)
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


class TbifAcquirer(FileAcquirer):
    def setup(self):
        if os.path.exists(self.filename):
            print("Looking for ", self.filename)
            _, ext = os.path.splitext(self.filename)[:2]
            if ext == ".tbif":
                self.data = []
                self.stim = []
                self.zpos = []
                with open(self.filename, mode="rb") as f:
                    data = f.read()
                    header = struct.unpack("=IdHHffffdd", data[0:48])
                    # res is: fpp, spf, w, h,
                    img_size = header[2] * header[3]
                    # -------- repeats here, starts at 48, read 4, 12, img_size
                    for i in range(0, 2878):
                        (zpos,) = struct.unpack(
                            "=f",
                            data[
                                48
                                + (16 + img_size * 2) * i : 52
                                + (16 + img_size * 2) * i
                            ],
                        )
                        self.zpos.append(zpos)
                        stim = struct.unpack(
                            "=fff",
                            data[
                                52
                                + (16 + img_size * 2) * i : 64
                                + (16 + img_size * 2) * i
                            ],
                        )
                        self.stim.append(np.asarray(stim))
                        img = struct.unpack(
                            "=" + str(img_size) + "H",
                            data[
                                64
                                + (16 + img_size * 2) * i : 64
                                + img_size * 2
                                + (16 + img_size * 2) * i
                            ],
                        )
                        tmp = np.reshape(
                            np.asarray(img, dtype="uint16"),
                            (header[2], header[3]),
                            order="F",
                        )
                        self.data.append(tmp.transpose())
                    self.data = np.array(self.data)
            else:
                logger.error("Cannot load file, bad extension")
                raise Exception
        else:
            raise FileNotFoundError

    def runAcquirer(self):
        t = time.time()

        if self.done:
            pass
        elif self.frame_num < len(self.data):
            frame = self.getFrame(self.frame_num)
            if self.frame_num == len(self.data):
                print("Done with dataset ", self.frame_num)
            id = self.client.put(frame, "acq_raw" + str(self.frame_num))
            self.timestamp.append([time.time(), self.frame_num])
            try:
                self.q_out.put([{str(self.frame_num): id}])
                self.links["stim_queue"].put(
                    {self.frame_num: self.stim[self.frame_num % len(self.stim)]}
                )
                # logger.info('Current stim: {}'.format(self.stim[self.frame_num]))
                self.frame_num += 1
                # self.saveFrame(frame) # Also log to disk
            except Exception as e:
                logger.error("Acquirer general exception: {}".format(e))

            time.sleep(self.framerate)  # pretend framerate

        else:
            logger.error("Done with all available frames: {0}".format(self.frame_num))
            self.data = None
            self.q_comm.put(None)
            self.done = True

        self.total_times.append(time.time() - t)

    def getFrame(self, num):
        """Here just return frame from loaded data"""
        return self.data[num, 30:470, :]
