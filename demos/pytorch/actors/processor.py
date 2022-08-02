import time
import numpy as np
from improv.store import CannotGetObjectError, ObjectNotFoundError
from queue import Empty
from improv.actor import Actor, RunManager

import torch
# import torch.multiprocessing as mp

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import logging; logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
import traceback

class PyTorchProcessor(Actor):
    # TODO: Update ALL docstrings
    # TODO: Clean commented sections
    # TODO: add GPU/CPU option as input...
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ''' 
    '''

    # def __init__(self, *args,  transforms_path=None, **kwargs):
    def __init__(self, *args,  transforms_path=None, model_path=None, label_path=None, classify=True, **kwargs):
        super().__init__(*args)
        logger.info(transforms_path)
        if transforms_path is None:
            # logger.error("Must specify a transforms path.")
            logger.error("Must specify a JIT-compiled transforms path.")
        else:
            self.transforms_path = transforms_path
        if model_path is None:
            # logger.error("Must specify a transforms path.")
            logger.error("Must specify a JIT-compiled model path.")
        else:
            self.model_path = model_path
        if classify is True and label_path is None:
            logger.error("Must specify a path to image labels.")
        else:
            self.label_path = label_path

        self.img_num = 0

        # Add cuda:0 ect. as input - device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # if classify is True:
        #     self.classify = True

    def setup(self):
        ''' Initialize transforms model
        '''
        logger.info('Loading transforms for ' + self.name)
        self.done = False
        self.dropped_img = []

        # try:
        #     mp.set_start_method("forkserver")
        # except: pass

        t = time.time()

        # self.transforms = torch.jit.load(self.transforms_path).to(self.device)

        self.transforms = torch.jit.load(self.transforms_path).to(self.device).share_memory
        
        self.load_transforms_time = time.time() - t
        print('Time to load transforms: ', self.load_transforms_time*1000.0)
        with open("output/timing/load_transforms_time.txt", "w") as text_file:
            text_file.write("%s" % self.load_transforms_time)
            
        logger.info('Loading model for ' + self.name)

        t = time.time()

        # import os
        # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

        # self.model = torch.jit.load(self.model_path).to(self.device)

        self.model = torch.jit.load(self.model_path).to(self.device).share_memory

        # num_processes = 4
        # processes = []
        # for rank in range(num_processes):
        #     p = mp.Process(target=self.transforms, args=(self.model,))
        #     p.start()
        #     processes.append(p)
        # for p in processes:
        #     p.join()

        self.load_model_time = time.time() - t
        print('Time to load model: ', self.load_model_time*1000.0)
        with open("output/timing/load_model_time.txt", "w") as text_file:
            text_file.write("%s" % self.load_model_time)

        # return self.transforms, self.model

    def run(self):
        ''' Run the processor continually on input data, e.g.,images
        '''
        self.img_num = []
        self.get_img_out = []
        self.proc_img_time = []
        self.put_img_out = []
        self.inference_time = []
        self.put_out_time = []
        self.pred_time = []
        self.put_pred_time = []
        self.total_times = []
        
        with RunManager(self.name, self.runProcess, self.setup, self.q_sig, self.q_comm) as rm:
            print(rm)
        
        print('Processor broke, avg time per image: ', np.mean(self.total_times, axis=0))
        print('Processor got through ', self.img_num, ' images')

        np.savetxt('output/timing/get_img_out.txt', np.array(self.get_img_out))
        np.savetxt('output/timing/process_image_time.txt', np.array(self.proc_img_time))
        np.savetxt('output/timing/put_image_out.txt', np.array(self.put_img_out))
        np.savetxt('output/timing/inference_time.txt', np.array(self.inference_time))
        np.savetxt('output/timing/put_out_time.txt', np.array(self.put_out_time))
        if self.classify is True:
            np.savetxt('output/timing/prediction_time.txt', np.array(self.pred_time))
            np.savetxt('output/timing/put_prediction_time.txt', np.array(self.put_pred_time))
        np.savetxt('output/timing/total_times.txt', np.array(self.total_times))

    def runProcess(self):
        ''' Run process. Runs once per image.
            Output is a location in the DS to continually
            place the processed image, model output, and classification/prediction with ref number that
            corresponds to the frame number (TODO)
            [From neurofinder/actors/processor.py]
        '''
        ids = self._checkImages()

        if ids is not None:
            t = time.time()
            self.done = False
            try:
                # Maybe switch ids 0 and 1?
                img = self.client.getID(ids[0])
                t1 = time.time()
                img = self._processImage(img)
                t2 = time.time()
                img_obj_id = self.client.put(img, 'proc_img' + str(self.img_num))
                self.q_out.put([img_obj_id, str(self.img_num)])
                output = self._runInference(img)
                t3 = time.time()
                out_obj_id = self.client.put(output.numpy(), 'output' + str(self.img_num))
                self.q_out.put([[out_obj_id, str(self.img_num)]])
                t4 = time.time()
                if self.classify is True:
                    pred = self._classifyImage(output, self.img_num)
                    t5 = time.time()
                    pred_obj_id = self.client.put(pred, 'prediction' + str(self.img_num))
                    # self.put([[pred_obj_id, str(self.img_num)]], save=True)
                    t6 = time.time()
                    self.pred_time.append((t5 - t4)*1000.0)
                    self.put_pred_time.append((t6 - t5)*1000.0)

                self.get_img_out.append((t1 - t)*1000.0)
                self.proc_img_time.append((t2 - t1)*1000.0)
                self.put_img_out.append((t3 - t2)*1000.0)
                self.inference_time.append((t3 - t2)*1000.0)
                self.put_out_time.append((t4 - t3)*1000.0)

                self.img_num += 1
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
            self.total_times.append(time.time()-t)
        else:
            pass

# Should the following technically be internal?
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

    def _processImage(self, img):
        ''' Load data - here, .jpg image to tensor
        Input is already loaded image from q_in
        TODO: Time? Above?
        '''
        # t = time.time()
        if img is None:
            raise ObjectNotFoundError
        img = self.transforms(img).unsqueeze(dim=0)
        return img

        # self.load_img_time.append([time.time() - t])

    def _runInference(self, data):
        '''
        '''
        data = data.to(self.device)
        with torch.no_grad():
            output = self.model(data).squeeze(dim=0).to(self.device)
        return output

    # def _classifyImage(self, labels=('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')):
    #     '''
    #     '''
    #     _, predicted = torch.max(self.output.data, 1)

    #     with open(self.label_path + "label/{}.txt".format(self.img_num), "r") as text_file:
    #         text_file.read("%s" % self.load_transforms_time)

    #     correct = (predicted == labels).sum().item()

class NaNDataException(Exception):
    pass








   

  
