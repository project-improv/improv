import numpy as np
import os
import pandas as pd
from pathlib import Path # for FolderAcquirer
from skimage.io import imread
import time

from improv.actor import Actor

import logging; logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class ImageAcquirer(Actor):
    # TODO: Update ALL docstrings
    # TODO: Clean commented sections
    # TODO: Add any relevant q_comm
    ''' Acquire all image files in a folder.
        Reads TIFF/JPG/PNG files in a specified {path} from disk.
    '''

    def __init__(self, *args, n_imgs=None, folder=None, exts=None, classify=False, label_folder=None, time_opt=False, timing=None, lab_timing=None, out_path=None, **kwargs):
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
        
        self.time_opt = time_opt
        if self.time_opt is True:
            self.timing = timing
            self.lab_timing = lab_timing
            self.out_path = out_path

        self.n_imgs = n_imgs

    def setup(self):
        os.makedirs(self.out_path, exist_ok=True)

        self.acq_total_times = []
        self.acq_timestamps = []
        self.get_img_time = []
        self.put_img_time = []
        self.put_out_time = []
        self.timestamps = []

        if self.classify is True and self.label_path is not None:
            self.get_lab_time = []
            self.put_lab_time = []
            self.lab_timestamps = []

        self.files = [f.as_posix() for f in self.path.iterdir() if f.suffix in self.exts]
        self.files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

        if self.classify is True and self.label_path is not None:
            self.label_files = [f.as_posix() for f in self.label_path.iterdir()]
            self.label_files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
        
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

    def runStep(self):
        ''' Triggered at Run
            Get list of files in the folder and use that as the baseline.
            Arg: exts = list of possible extensions
        '''

        # if self.time is True:
        #     for l in self.timing:
        #         setattr(self, l, [])
        #         # Equivalent to:
        #         self.acq_timestamps = []
        #         self.put_img_time = []
        #         self.put_out_time = []
        #         self.timestamps = []
                # values = [self.put_img_time, self.acq_timestamps, self.put_out_time, self.acq_total_times] 
        
    # for k, v in kwargs.items():
    #     setattr(self, k, v)

        # if self.classify is True and self.label_path is not None:
        #     for l in self.lab_timing:
        #         setattr(self, l, [])
        #         Equivalent to:
        #         self.put_lab_time = []
        #         self.lab_timestamps = []


        '''
        '''
        if self.time_opt is True:
            self.acq_timestamps.append((time.time(), int(self.img_num)))

        if self.done:
            pass
        
        elif self.img_num != self.n_imgs:
            t = time.time()
            try:
                t1 = time.time()
                img = self.get_img(self.files[self.img_num])
                self.timestamps.append((time.time()*1000.0, self.img_num))
                t2 = time.time()
                img_obj_id = self.client.put(img, 'acq_raw' + str(self.img_num))
                t3 = time.time()
                if self.classify is True and self.label_path is not None:
                    lab = self.get_label(self.label_files[self.img_num])
                    self.lab_timestamps.append((time.time()*1000.0, self.img_num))
                    t4 = time.time()
                    lab_obj_id = self.client.put(lab, 'acq_lab' + str(self.img_num))
                    t5 = time.time()
                    self.q_out.put([img_obj_id, lab_obj_id, str(self.img_num)])
                    self.put_out_time.append((time.time() - t5)*1000.0)
                    self.get_lab_time.append((t4 - t3)*1000.0)
                    self.put_lab_time.append((t5 - t4)*1000.0)
                else:
                    self.q_out.put([img_obj_id, str(self.img_num)])
                    self.put_out_time.append((time.time() - t3)*1000.0)
         
                self.img_num += 1
                self.acq_total_times.append((time.time() - t)*1000.0)
                self.get_img_time.append((t2 - t1)*1000.0)
                self.put_img_time.append((t3 - t2)*1000.0)

                # if self.time is True:
                #     for l in self.timing:
                #             setattr(self, l, [])
                #             # Equivalent to:
                #             self.acq_timestamps = []
                #             self.put_img_time = []
                #             self.put_out_time = []
                #             self.timestamps = []
                # # values = [self.put_img_time, self.acq_timestamps, self.put_out_time, self.acq_total_times] 
                    
                # for k, v in kwargs.items():                 

            except Exception as e:
                logger.error('Acquirer general exception: {}'.format(e))
            except IndexError as e:
                pass

        if self.img_num == self.n_imgs:
            logger.error('Done acquiring all available data: {}'.format(self.img_num))
            self.data = None
            self.q_comm.put(None)
            self.done = True


        if '.tif' or '.tiff' in self.exts:
            print('Acquire broke, avg time per frame:', np.mean(self.acq_total_times))
            print('Acquire got through', self.img_num, ' frames')
        if '.jpg' or '.png' in self.exts:
            print('Acquire broke, avg time per image:', np.mean(self.acq_total_times))
            print('Acquire got through ', self.img_num, ' images')

    def stop(self):
        '''
        '''

        if self.time_opt is True:
            keys = self.timing
            values = [self.acq_timestamps, self.get_img_time, self.put_img_time, self.put_out_time, self.acq_total_times] 

            if self.classify is True and self.label_path is not None:
                keys.extend(self.lab_timing)
                values.extend([self.get_lab_time, self.put_lab_time, self.lab_timestamps])
            
            timing_dict = dict(zip(keys, values))
            df = pd.DataFrame.from_dict(timing_dict, orient='index').transpose()
            df.to_csv(os.path.join(self.out_path, 'acq_timing_' + str(self.n_imgs) + '.csv'), index=False, header=True)

        return 0

    def get_img(self, file):
        '''
        '''
        try:
            img = imread(file)
        except ValueError as e:
            img = imread(file)
            logger.error('File ' + file + ' had value error {}'.format(e))
        return img

    def get_label(self, label_file):
        '''
        TODO: User input -> label str
        '''
        with open(label_file, "r") as f:
            label = int(f.read())
            f.close()
        return label