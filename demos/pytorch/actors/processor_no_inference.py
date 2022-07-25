import time
import numpy as np
from improv.store import Limbo, CannotGetObjectError, ObjectNotFoundError
from queue import Empty
from improv.actor import Actor, RunManager

import torch

import logging; logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class PyTorchProcessor(Actor):
    # TODO: Update ALL docstrings
    # TODO: Clean commented sections
    # TODO: add GPU/CPU option as input...
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ''' 
    '''

    def __init__(self, *args, transforms_path=None):
        super().__init__(*args)
        logger.info(transforms_path)
        if transforms_path is None:
            # logger.error("Must specify a transforms path.")
            logger.error("Must specify a JIT-compiled transforms path.")
        else:
            self.transforms_path = transforms_path
        self.sample_number = 0

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def setup(self, message):
        ''' Initialize transforms model
        '''
        logger.info('Loading transforms for ' + self.name)
        self.done = False
        self.dropped_img = []

        t = time.time()

        self.transforms = torch.jit.load(self.transforms_path).to(self.device)
        
        self.load_model_time = time.time() - t
        print(message, self.load_model_time*1000.0)
        with open("output/timing/load_model_time.txt", "w") as text_file:
            text_file.write("%s" % self.load_model_time)

        return self.transforms

    def run(self):
        ''' Run the processor continually on input data, e.g.,images
        '''
        self.img_num = []
        self.load_img_time = []
        self.proc_img_time = []
        self.put_out_time = []
        self.timestamp = []
        self.total_times = []

        with RunManager(self.name, self.runProcess, self.setup, self.q_sig, self.q_comm) as rm:
            print(rm)
        
        print('Processor broke, avg time per image: ', np.mean(self.total_times, axis=0))
        print('Processor got through ', self.image_number, ' images')

        np.savetxt('output/timing/process_image_time.txt', np.array(self.proc_img_time))
        np.savetxt('output/timing/process_timestamp.txt', np.array(self.timestamp))

    def runProcess(self):
        ''' Run process. Runs once per image.
            Output is a location in the DS to continually
            place the processed image, with ref number that
            corresponds to the frame number (TODO)
            [From neurofinder/actors/processor.py]
        '''
        img = self._checkImages()

        if img is not None:
            t = time.time()
            self.done = False
            try:
                self.img = self.client.getID([str(self.img_num)])
                t1 = time.time()
                self.get_img_out.appennd(t1 - t)
                self.img = self._processImage(self.img, self.img_num)
                t2 = time.time()
                self.proc_img_time.append([t1 - t2])                
                self.timestamp.append([time.time(), self.img_num])
            # Insert exceptions here...ERROR HANDLING, SEE ANNE'S ACTORS - from 1p demo
            except ObjectNotFoundError:
                logger.error('Processor: Image {} unavailable from store, dropping'.format(self.img_num))
                self.dropped_img.append(self.img_num)
                self.q_out.put([1])
            except KeyError as e:
                logger.error('Processor: Key error... {0}'.format(e))
                # Proceed at all costs
                self.dropped_img.append(self.img_num)
            except Exception as e:
                logger.error('Processor error: {}: {} during image number {}'.format(type(e).__name__,
                                                                                            e, self.frame_number))
                print(traceback.format_exc())
                self.dropped_img.append(self.img_num)
            self.img_num += 1
            self.total_times.append(time.time()-t)
        else:
            pass

    def _checkImages(self):
        ''' Check to see if we have images for processing
        From basic demo
        '''
        # t = time.time()
        try:
            # Timeout = 0 ms
            res = self.q_in.get(timeout=0)
            return res
        #TODO: additional error handling
        except Empty:
            logger.info('No images for processing')
            return None
        # self.get_in_time = time.time() - t

    def _processImage(self, img, img_num):
        ''' Load data - here, .jpg image to tensor
        Input is already loaded image from q_in
        TODO: Time? Above?
        '''
        # t = time.time()
        if img is None:
            raise ObjectNotFoundError
        img = self.transforms(self.q_in).unsqueeze(dim=0).to(self.device)
        return img

        # self.load_img_time.append([time.time() - t])

class NaNDataException(Exception):
    pass





    

