import time
import os
import h5py
import random
import numpy as np
from skimage.io import imread

# For FolderAcquirer
from pathlib import Path
from queue import Empty

from improv.actor import Actor, RunManager

import logging; logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Modified from ~/improv/demos/neurofinder/actors/acquire_folder.py
# Maybe rename - ImageAcquirer? 
# Moved to src
class FolderAcquirer(Actor):
    ''' Current behavior is looping over all files in a folder.
        Class to read TIFF/JPG/PNG files in a specified {path} from disk.
        Designed for scenarios when new TIFF files are created during the run.
        Reads only new TIFF files (after Run started) and put on to the Plasma store.
        If there're multiple files, files are loaded by name.
    '''

    def __init__(self, *args, folder=None, exts=None, classify=False, label_folder=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.done = False

        self.sample_num = 0

        self.classify = classify

        if folder is None:
            logger.error('Must specify folder of data.')
        else:
            self.files = []
            self.path = Path(folder)
            if not self.path.exists() or not self.path.is_dir():
                raise AttributeError('Data folder {} does not exist.'.format(self.path))

        if exts is None:
            logger.error('Must specify filetype/(possible) extension(s) of data.')
        else:
            self.exts = exts

        if self.classify is True and label_folder is None:
            logger.error('Must specify folder of labels.')
        else:
            self.lab_files = []
            self.lab_path = Path(label_folder)
            if not self.lab_path.exists() or not self.lab_path.is_dir():
                raise AttributeError('Label folder {} does not exist.'.format(self.lab_path))

    def setup(self):
        pass
        
    # def saveImgs(self):
    #     '''
    #     Arg: exts = list of possible file extensions
    #     '''
    #     self.imgs = []
    #     files = [f.as_posix() for f in self.path.iterdir() if f.suffix in self.exts]
    #     files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    #     for file in files:
    #         img = self.get_sample(file)
    #         self.imgs.append(img)
    #     self.imgs = np.array(self.imgs)
    #     f = h5py.File('output/sample.h5', 'w', libver='latest')
    #     f.create_dataset("default", data=self.imgs)
    #     f.close()

    def run(self):
        ''' Triggered at Run
            Get list of files in the folder and use that as the baseline.
            Arg: exts = list of possible extensions
        '''
        self.put_img_time = []
        self.total_times = []
        self.timestamp = []

        self.files = [f.as_posix() for f in self.path.iterdir() if f.suffix in self.exts]
        self.files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
        
        if self.classify is True:
            self.put_lab_time = []
            self.lab_timestamp = []
            self.lab_files = [f.as_posix() for f in self.lab_path.iterdir()]
            self.lab_files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

        with RunManager(self.name, self.runAcquirer, self.setup, self.q_sig, self.q_comm) as rm:
            print(rm)

        if '.tif' or '.tiff' in self.exts:
            print('Acquire broke, avg time per frame: ', np.mean(self.total_times))
            print('Acquire got through ', self.sample_num, ' frames')
        if '.jpg' or '.png' in self.exts:
            print('Acquire broke, avg time per image: ', np.mean(self.total_times))
            print('Acquire got through ', self.sample_num, ' images')

        np.savetxt('output/timing/put_image_time.txt', np.array(self.put_img_time))
        np.savetxt('output/timing/acquire_timestamp.txt', np.array(self.timestamp))
        
        if self.classify is True:
            np.savetxt('output/timing/put_label_time.txt', np.array(self.put_lab_time))
            np.savetxt('output/timing/acquire_lab_timestamp.txt', np.array(self.lab_timestamp))

    def runAcquirer(self):
        ''' Main loop. If there're new files, read and put into store.
        '''
        if self.done:
            pass
        
        else:
            t = time.time()
            try:
                t1 = time.time()
                obj_id = self.client.put(self.files[self.sample_num], 'acq_raw' + str(self.sample_num))
                self.timestamp.append([time.time(), self.sample_num])
                self.q_out.put([obj_id, str(self.sample_num)])
                self.put_img_time = time.time() - t1
                # Get lab at same time as image? Simulate human labeling image?
                if self.classify is True:
                    t2 = time.time()
                    obj_id = self.client.put(self.lab_files[self.sample_num], 'acq_lab' + str(self.sample_num))
                    self.lab_timestamp.append([time.time(), self.sample_num])
                    self.q_out.put([obj_id, str(self.sample_num)])
                    self.put_lab_time = time.time() - t2
                self.sample_num += 1
            except Exception as e:
                logger.error('Acquirer general exception: {}'.format(e))
            except IndexError as e:
                pass

            # time.sleep(self.framerate)  # pretend framerate
            self.total_times.append(time.time() - t)
        
        # From bubblewrap demo and FileAcquirer
        logger.error('Done with all available data: {0}'.format(self.sample_num))
        self.data = None
        self.q_comm.put(None)
        self.done = True  # stay awake in case we get a shutdown signal

    # def get_sample(self, file):
    #     '''
    #     '''
    #     try:
    #         img = imread(file)
    #     except ValueError as e:
    #         img = imread(file)
    #         logger.error('File '+file+' had value error {}'.format(e))
    #     # For .tiff: [0,0,0, :, :,0] - extract first channel in image set
    #     # For .jpg/.png -> PyTorch, tensor in processor
    #     return img