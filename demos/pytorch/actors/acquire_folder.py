import time
import os
# import h5py
# import random
import numpy as np
from skimage.io import imread
import json

# For FolderAcquirer
from pathlib import Path
from queue import Empty

from improv.actor import Actor, RunManager

import logging; logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# FOR TIMING:
# t = time.time()
# [some process]
# torch.cuda.current_stream().synchronize()
# t = time.time()
# WHICH IS BETTER???

# Modified from ~/improv/demos/neurofinder/actors/acquire_folder.py
# Maybe rename - ImageAcquirer? 
class FolderAcquirer(Actor):
    ''' Current behavior is looping over all files in a folder.
        Class to read TIFF/JPG/PNG files in a specified {path} from disk.
        Designed for scenarios when new TIFF files are created during the run.
        Reads only new TIFF files (after Run started) and put on to the Plasma store.
        If there're multiple files, files are loaded by name.
    '''

    def __init__(self, *args, sleep_time=.5, folder=None, exts=None, classify=False, label_folder=None, out_path=None, **kwargs):
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
            logger.error('Must specify filetype/(possible) extension(s) of data.')
        else:
            self.exts = exts

        if self.classify is True and label_folder is None:
            logger.error('Must specify folder of labels.')
        elif self.classify is True and label_folder is not None:
            self.lab_path = Path(label_folder)
            if not self.lab_path.exists() or not self.lab_path.is_dir():
                raise AttributeError('Label folder {} does not exist.'.format(self.lab_path))
        
        self.out_path = out_path

        self.sleep_time = sleep_time

    def setup(self):
        os.makedirs(self.out_path, exist_ok=True)

        self.files = [f.as_posix() for f in self.path.iterdir() if f.suffix in self.exts]
        self.files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

        if self.classify is True and self.lab_path is not None:
            self.lab_files = [f.as_posix() for f in self.lab_path.iterdir()]
            self.lab_files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

        self.n_imgs = len(self.files)
        
    # def saveImgs(self):
    #     '''
    #     Arg: exts = list of possible file extensions
    #     '''
    #     self.imgs = []
    #     files = [f.as_posix() for f in self.path.iterdir() if f.suffix in self.exts]
    #     files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    #     for file in files:
    #         img = self.get_img(file)
    #         self.imgs.append(img)
    #     self.imgs = np.array(self.imgs)
    #     f = h5py.File('output/img.h5', 'w', libver='latest')
    #     f.create_dataset("default", data=self.imgs)
    #     f.close()

    def run(self):
        ''' Triggered at Run
            Get list of files in the folder and use that as the baseline.
            Arg: exts = list of possible extensions
        '''
        self.put_img_time = []
        self.put_out_time = []
        self.timestamp = []
        self.acq_total_times = []

        if self.classify is True and self.lab_path is not None:
            self.put_lab_time = []
            self.lab_timestamp = []

        with RunManager(self.name, self.runAcquirer, self.setup, self.q_sig, self.q_comm) as rm:
            print(rm)

        # if self.img_num == 200:
        #     with open(self.out_path + "acq_vars.txt", "w") as f:
        #         print(vars(self), file=f)
        #         f.close()

        if '.tif' or '.tiff' in self.exts:
            print('Acquire broke, avg time per frame: ', np.mean(self.acq_total_times))
            print('Acquire got through ', self.img_num, ' frames')
        if '.jpg' or '.png' in self.exts:
            print('Acquire broke, avg time per image: ', np.mean(self.acq_total_times))
            print('Acquire got through ', self.img_num, ' images')

        # np.savetxt(self.out_path + 'put_img_time.txt', np.array(self.put_img_time))
        # np.savetxt(self.out_path + 'acquire_timestamp.txt', np.array(self.timestamp))
        # np.savetxt(self.out_path + 'put_out_time.txt', np.array(self.put_out_time))
        # logger.error('ACQ1: TESTTESTTEST!!!')

        # if self.classify is True and self.lab_path is not None:
        #     np.savetxt(self.out_path + 'put_lab_time.txt', np.array(self.put_lab_time))
        #     np.savetxt(self.out_path + 'acquire_lab_timestamp.txt', np.array(self.lab_timestamp))
        #     logger.error('ACQ2: TESTTESTTEST!!!')

        # np.savetxt(self.out_path + 'acq_total_times.txt', np.array(self.acq_total_times))

        print('Done saving out!')
        
    def runAcquirer(self):
        ''' Main loop. If there're new files, read and put into store.
        '''
        if self.done:
            pass
        
        elif self.img_num != len(self.files):
            t = time.time()
            try:
                t1 = time.time()
                img_obj_id = self.client.put(self.get_img(self.files[self.img_num]), 'acq_raw' + str(self.img_num))
                self.timestamp.append([time.time()*1000.0, self.img_num])
                self.put_img_time.append((time.time() - t1)*1000.0)
                t2 = time.time()
            # Get lab at same time as image? Simulate human labeling image?
                if self.classify is True and self.lab_path is not None:
                    t3 = time.time()
                    lab_obj_id = self.client.put(self.get_label(self.lab_files[self.img_num]), 'acq_lab' + str(self.img_num))
                    self.put_lab_time.append((time.time() - t3)*1000.0)
                    self.lab_timestamp.append([time.time()*1000.0, self.img_num])
                    t4 = time.time()
                    self.q_out.put([img_obj_id, lab_obj_id, str(self.img_num), self.n_imgs])
                    self.put_out_time.append((time.time() - t4)*1000.0)
                else:
                    self.q_out.put([img_obj_id, str(self.img_num), self.n_imgs])
                    self.put_out_time.append((time.time() - t2)*1000.0)
         
                self.img_num += 1
                self.acq_total_times.append((time.time() - t)*1000.0)

                if self.img_num == self.n_imgs:
                    tmp = vars(self)
                    with open(self.out_path + "acq_vars.txt", "w+") as f:
                        d = {str(item): str(tmp[item]) for item in tmp}
                        print(d, file=f)
                        # print(vars(self), file=f)
                        f.close()
                        # print('Done saving!')

            except Exception as e:
                logger.error('Acquirer general exception: {}'.format(e))
            except IndexError as e:
                pass

        # From bubblewrap demo and FileAcquirer
        if self.img_num == self.n_imgs:
            logger.error('Done acquiring all available data: {}'.format(self.img_num))
            self.data = None
            self.q_comm.put(None)
            self.done = True  # stay awake in case we get a shutdown signal

    def get_img(self, file):
        '''
        '''
        try:
            img = imread(file)
        except ValueError as e:
            img = imread(file)
            logger.error('File ' + file + ' had value error {}'.format(e))
        # For .tiff: [0,0,0, :, :,0] - extract first channel in image set
        # For .jpg/.png -> PyTorch, tensor in processor
        return img

    def get_label(self, label_file):
        '''
        TODO: Maybe can make one line? Speed up? User input -> label str
        '''
        with open(label_file, "r") as f:
            label = int(f.read())
            f.close()
        return label