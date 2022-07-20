import time
import os
# import h5py
import numpy as np

from pathlib import Path
from improv.actor import Actor, RunManager
from queue import Empty

from PIL import Image
import torchvision.transforms as transforms


import logging; logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Modified from ~/improv/demos/neurofinder/actors

class FolderAcquirer(Actor):
    ''' Current behavior is looping over all files in a folder.
        Class to read JPG files in a specified {path} from disk.
        Designed for scenarios when new JPG files are created during the run.
        Reads only new JPG files (after Run started) and put on to the Plasma store.
        If there're multiple files, files are loaded by name.
    '''

    def __init__(self, *args, folder=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.data = None
        self.done = False
        self.path = Path(folder)

        self.img_num = 0
        self.files= []

        # Estimated time to get image from device
        self.lag = 0.005

        if not self.path.exists() or not self.path.is_dir():
            raise AttributeError('Folder {} does not exist.'.format(self.path))

    def setup(self):
        pass
        
    # Save image data acquires as h5py - not necessary
    # def saveImgs(self):
    #     self.imgs = []
    #     files = {f for f in self.path.iterdir() if f.suffix in ['.jpg', '.png']}
    #     files = sorted(list(files))
    #     for file in files:
    #         img = self.get_img(file)
    #         self.imgs.append(img)
    #     self.imgs = np.array(self.imgs)
    #     f = h5py.File('output/sample.h5', 'w', libver='latest')
    #     f.create_dataset("default", data=self.imgs)
    #     f.close()

    def run(self):
        ''' Triggered at Run
            Get list of files in the folder and use that as the baseline.
        '''
        self.total_times = []
        self.timestamp = []

        self.files = [f for f in self.path.iterdir() if f.suffix in ['.jpg', '.png']]

        with RunManager(self.name, self.runAcquirer, self.setup, self.q_sig, self.q_comm) as rm:
            print(rm)

        print('Acquire broke, avg time per frame: ', np.mean(self.total_times))
        print('Acquire got through ', self.img_num, ' images')

    def runAcquirer(self):
        ''' Main loop. If there're new files, read and put into store.
        Adapted from bubblewrap demo
        '''
        
        if self.done:
            pass  

        elif:
            t = time.time()
            try:
                obj_id = self.client.put(self.get_img(self.files[self.img_num]), 'acq_raw_img' + str(self.img_num))
                self.timestamp.append([time.time(), self.img_num])
                self.q_out.put([[obj_id, str(self.img_num)]], save=[True])
                self.img_num += 1
            except Exception as e:
                logger.error('Acquirer general exception: {}'.format(e))
            except IndexError as e:
                pass
                
            time.sleep(self.lag)
            self.total_times.append(time.time() - t)

        # From bubblewrap demo
        else:  # simulating a done signal from the source
            logger.error('Done with all available images: {0}'.format(self.img_num))
            self.data = None
            self.q_comm.put(None)
            self.done = True  # stay awake in case we get a shutdown signal

    def get_img(self, file: Path):
        try:
            img = Image.open(file.as_posix())
            # transform = transforms.Compose([transforms.PILToTensor()])
            # img_tensor = transform(img)
        except ValueError as e:
            img = Image.open(file.as_posix())
            logger.error('File '+file.as_posix()+' had value error {}'.format(e))
        return img