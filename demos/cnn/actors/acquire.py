import time
import os
# import h5py
# import random
import numpy as np
import pandas as pd
from skimage.io import imread

# For FolderAcquirer
from pathlib import Path
from queue import Empty

from improv.actor import Actor, RunManager

import logging; logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class ImageAcquirer(Actor):
    ''' Current behavior is looping over all image files in a folder.
        Class to read TIFF/JPG/PNG files in a specified {path} from disk.
        Designed for scenarios when new TIFF/JPG/PNG files are created during the run.
        Reads only new TIFF/JPG/PNG files (after Run started) and put on to the Plasma store.
        If there're multiple files, files are loaded by name.
    '''

    def __init__(self, *args, n_imgs=None, folder=None, exts=None, classify=False, label_folder=None, out_path=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.done = False

        self.img_num = 0

        self.classify = classify

        if folder is None:
            logger.error('Must specify folder of data.')
        else:
            self.path = Path(folder)
            if not self.path.exists() or not self.path.is_dir():
                raise AttributeError('Data folder {} does not exist.'.format(self.path))

        if exts is None:
            logger.error('Must specify filetype/extension(s) of data.')
        else:
            self.exts = exts

        if self.classify is True and label_folder is None:
            logger.error('Must specify folder of labels.')
        elif self.classify is True and label_folder is not None:
            self.label_path = Path(label_folder)
            if not self.label_path.exists() or not self.label_path.is_dir():
                raise AttributeError('Label folder {} does not exist.'.format(self.label_path))

        self.n_imgs = n_imgs

        self.out_path = out_path

    def setup(self):
        '''
        '''
        os.makedirs(self.out_path, exist_ok=True)

        self.files = [f.as_posix() for f in self.path.iterdir() if f.suffix in self.exts]
        self.files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

        if self.classify is True and self.label_path is not None:
            self.label_files = [f.as_posix() for f in self.label_path.iterdir()]
            self.label_files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
            
    def run(self):
        ''' Triggered at Run
            Get list of files in the folder and use that as the baseline.
            Arg: exts = list of possible extensions
        '''
        self.acq_total_times = []
        
        with RunManager(self.name, self.runAcquirer, self.setup, self.q_sig, self.q_comm) as rm:
            print(rm)

        if '.tif' or '.tiff' in self.exts:
            print('Acquire broke, avg time per frame:', np.mean(self.acq_total_times))
            print('Acquire got through', self.img_num, ' frames')
        if '.jpg' or '.png' in self.exts:
            print('Acquire broke, avg time per image:', np.mean(self.acq_total_times))
            print('Acquire got through ', self.img_num, ' images')

    def runAcquirer(self):
        ''' Main loop. If there're new files, read and put into store.
        '''
        if self.done:
            pass
        
        elif self.img_num != self.n_imgs:
            t = time.time()
            try:
                img = self.get_img(self.files[self.img_num])
                img_obj_id = self.client.put(img, 'acq_raw' + str(self.img_num))
                if self.classify is True and self.label_path is not None:
                    label = self.get_label(self.label_files[self.img_num])
                    lab_obj_id = self.client.put(label, 'acq_lab' + str(self.img_num))
                    self.q_out.put([img_obj_id, lab_obj_id, str(self.img_num)])
                else:
                    self.q_out.put([img_obj_id, str(self.img_num)])
         
                self.img_num += 1
                self.acq_total_times.append((time.time() - t)*1000.0)

            except Exception as e:
                logger.error('Acquirer general exception: {}'.format(e))
            except IndexError as e:
                pass

        if self.img_num == self.n_imgs:
            logger.error('Done acquiring all available data: {}'.format(self.img_num))
            self.data = None
            self.q_comm.put(None)
            self.done = True  # stay awake in case we get a shutdown signal

    def get_img(self, file):
        '''
        '''
        try:
            # Why does file.as_posix not work?
            img = imread(file)
        except ValueError as e:
            img = imread(file)
            logger.error('File ' + file + ' had value error {}'.format(e))
        return img

    def get_label(self, label_file):
        with open(label_file, "r") as f:
            label = int(f.read())
            f.close()
        return label