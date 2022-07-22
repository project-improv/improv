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

    def __init__(self, *args, folder=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.data = None
        self.done = False
        self.path = Path(folder)

        # self.frame_num = 0
        self.sample_num = 0
        self.files = []

        if not self.path.exists() or not self.path.is_dir():
            raise AttributeError('Folder {} does not exist.'.format(self.path))

    def setup(self):
        pass
        
    def saveImgs(self, exts):
        '''
        Arg: exts = list of possible file extensions
        '''
        self.imgs = []
        files = {f for f in self.path.iterdir() if f.suffix in exts}
        files = sorted(list(files))
        for file in files:
            img = self.get_sample(file)
            self.imgs.append(img)
        self.imgs = np.array(self.imgs)
        f = h5py.File('output/sample.h5', 'w', libver='latest')
        f.create_dataset("default", data=self.imgs)
        f.close()

    def run(self, exts):
        ''' Triggered at Run
            Get list of files in the folder and use that as the baseline.
            Arg: exts = list of possible extensions
        '''
        self.put_img_time = []
        self.total_times = []
        self.timestamp = []
        
        self.files = [f for f in self.path.iterdir() if f.suffix in exts]

        with RunManager(self.name, self.runAcquirer, self.setup, self.q_sig, self.q_comm) as rm:
            print(rm)

        if '.tif' or '.tiff' in exts:
            print('Acquire broke, avg time per frame: ', np.mean(self.total_times))
            print('Acquire got through ', self.sample_num, ' frames')
        if '.jpg' or '.png' in exts:
            print('Acquire broke, avg time per image: ', np.mean(self.total_times))
            print('Acquire got through ', self.sample_num, ' images')

    def runAcquirer(self):
        ''' Main loop. If there're new files, read and put into store.
        '''
        if self.done:
            pass
        
        else:
            t = time.time()
            try:
                t1 = time.time()
                obj_id = self.client.put(self.get_sample(self.files[self.sample_num]), 'acq_raw' + str(self.sample_num))
                self.timestamp.append([time.time(), self.sample_num])
                self.put([[obj_id, str(self.sample_num)]], save=[True])
                self.sample_num += 1
                self.put_img_time = time.time() - t1
            except Exception as e:
                logger.error('Acquirer general exception: {}'.format(e))
            except IndexError as e:
                pass

            self.total_times.append(time.time() - t)
        
        # From bubblewrap demo and FileAcquirer
        logger.error('Done with all available data: {0}'.format(self.sample_num))
        self.data = None
        self.q_comm.put(None)
        self.done = True  # stay awake in case we get a shutdown signal

    def get_sample(self, file: Path):
        self.load_img_time = []
        t = time.time()
        try:
            img = imread(file.as_posix())
            self.load_img_time.append(time.time() - t
        except ValueError as e:
            img = imread(file.as_posix())
            logger.error('File ' + file.as_posix() + ' had value error {}'.format(e))
        return img # [0,0,0, :, :,0]  # Extract first channel in this image set, or tensor/3D array for .jpg/.png

    # ONLY FOR NO_IMPROV
    def put_img(self, client, img, img_num):
        # print('Put image', t)
        # obj_id = self.client.put(self.get_img(self.files[self.img_num]), 'acq_raw_img' + str(self.img_num))
        # img = pickle.dumps(img_PIL, protocol=pickle.HIGHEST_PROTOCOL)
        if torch.is_tensor(img):
            obj_id = client.put(pickle.dumps(img, protocol=pickle.HIGHEST_PROTOCOL), 'acq_raw_img-' + str(img_num))
        else:
            obj_id = client.put(img, 'acq_raw_img-' + str(img_num))
        return obj_id
