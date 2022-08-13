import os
import time
import numpy as np
from improv.store import CannotGetObjectError, ObjectNotFoundError
from queue import Empty
from improv.actor import Actor, RunManager

from pyarrow._plasma import PlasmaObjectExists, ObjectNotAvailable, ObjectID

import torch

import logging; logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
import traceback

class PyTorchProcessor(Actor):
    # TODO: Update ALL docstrings
    # TODO: Clean commented sections
    # TODO: Add GPU/CPU option as input...
    # TODO: Add any relevant q_comm
    ''' 
    '''

    def __init__(self, *args, gpu=False, gpu_num=None, model_path=None, classify=False, labels=None, out_path=None, method='spawn', **kwargs):
        super().__init__(*args, **kwargs)
        logger.info(model_path)
        if model_path is None:
            # logger.error("Must specify a model path.")
            logger.error("Must specify a JIT-compiled model path.")
        else:
            self.model_path = model_path

        self.img_num = 0

        if gpu is True:
            self.device = torch.device("cuda:{}".format(gpu_num) if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")

        self.classify = classify

        if self.classify is True and labels is None:
            logger.error("Must specify labels for classification.")
        elif self.classify is True and labels is not None:
            self.labels = labels

        self.out_path = out_path

    def setup(self):
        ''' Initialize model
        '''
        # Idk where this should be?
        os.makedirs(self.out_path, exist_ok=True)
        logger.info('Loading model for ' + self.name)
        self.done = False
        self.dropped_img = []

        t = time.time()

        self.model = torch.jit.load(self.model_path).to(self.device)

        # self.model = torch.jit.load(self.model_path).to(self.device).share_memory

        load_model_time = time.time() - t
        print('Time to load model: ', load_model_time*1000.0)
        with open(self.out_path + "load_model_time.txt", "w") as text_file:
            text_file.write("%s" % load_model_time)

    def run(self):
        ''' Run the processor continually on input data, e.g.,images
        '''
        self.get_img_out = []
        self.proc_img_time = []
        self.to_device = []
        self.inference_time = []
        # self.out_to_np = []
        # self.put_out_store = []
        # self.put_q_out = []
        # if self.classify is True:
        #     self.pred_time = []
        #     self.put_pred_store = []
        #     self.put_pred_out = []

        self.total_times = []
        
        with RunManager(self.name, self.runProcess, self.setup, self.q_sig, self.q_comm, runStore=self._getStoreInterface()) as rm:
            print(rm)
        
        print('Processor broke, avg time per image: ', np.mean(self.total_times, axis=0))
        print('Processor got through ', self.img_num, ' images')

        np.savetxt(self.out_path + 'get_img_out.txt', np.array(self.get_img_out))
        np.savetxt(self.out_path + 'process_img_time.txt', np.array(self.proc_img_time))
        np.savetxt(self.out_path + 'to_device.txt', np.array(self.to_device))
        np.savetxt(self.out_path + 'inference_time.txt', np.array(self.inference_time))
        # np.savetxt(self.out_path + 'out_to_np.txt', np.array(self.out_to_np))
        # np.savetxt(self.out_path + 'put_out_store.txt', np.array(self.put_out_store))
        # np.savetxt(self.out_path + 'put_q_out.txt', np.array(self.put_q_out))
        # if self.classify is True:
        #     np.savetxt(self.out_path + 'prediction_time.txt', np.array(self.pred_time))
        #     np.savetxt(self.out_path + 'put_prediction_time.txt', np.array(self.put_pred_store))
        #     np.savetxt(self.out_path + 'put_prediction_time.txt', np.array(self.put_pred_out))
        np.savetxt(self.out_path + 'total_times.txt', np.array(self.total_times))

    def runProcess(self):
        ''' Run process. Runs once per image.
            Output is a location in the DS to continually
            place the processed image, model output, and classification/prediction with ref number that
            corresponds to the frame number (TODO)
            [From neurofinder/actors/processor.py]
        '''
        ids = self._checkInput()

        if ids is not None:
            t = time.time()
            self.done = False
            try:
                # Maybe switch ids 0 and 1?
                img = self.client.getID(ids[0])
                t1 = time.time()
                img = self._processImage(img)
                print(img)
                t2 = time.time()
                output, t_dev = self._runInference(img)
                print(t_dev)
                print(output)
                time.sleep(.01)
                t3 = time.time()
                # # Necessary? Time? Optimize storage?
                # img.detach()
                # output = output.detach().numpy()
                # t4 = time.time()
                # out_obj_id = self.client.put(output, 'output' + str(self.img_num))
                # t5 = time.time()
                # # if self.classify is False:
                # #     self.q_out.put([[out_obj_id, str(self.img_num)]])
                # t6 = time.time()
                # if self.classify is True:
                #     # Do number correct in visual? Or output w/pred?
                #     # Compare label in and label out, add to counter if correct, keep note of number for each class...but can compare list as both fill in, frame-by-frame? Updating confusion matrix?
                #     pred = self._classifyImage(output, self.labels)
                #     t7 = time.time()
                #     pred_obj_id = self.client.put(pred, 'prediction' + str(self.img_num))
                #     t8 = time.time()
                #     self.q_out.put([[out_obj_id, pred_obj_id, str(self.img_num)]])
                #     t9 = time.time()
                #     self.pred_time.append((t7 - t6)*1000.0)
                #     self.put_pred_store.append((t8 - t7)*1000.0)
                #     self.put_pred_out.append((t9 - t8)*1000.0)
                    # Visualize updated accuracy...update every few images? OR every image?
                    # Confusion matrix that updates w/incoming data

                self.get_img_out.append((t1 - t)*1000.0)
                self.proc_img_time.append((t2 - t1)*1000.0)
                self.to_device.append(t_dev*1000.0)
                self.inference_time.append((t3 - t2)*1000.0)
                # self.out_to_np.append((t4 - t3)*1000.0)
                # self.put_out_store.append((t5 - t4)*1000.0)
                # self.put_q_out.append((t6 - t5)*1000.0)

                self.img_num += 1
            # Insert exceptions here...ERROR HANDLING, SEE ANNE'S ACTORS - from 1p demo
            except ObjectNotFoundError:
                logger.error('Processor: Image {} unavailable from store, dropping'.format(self.img_num))
                self.dropped_img.append(self.img_num)
                # self.q_out.put([1])
            except KeyError as e:
                logger.error('Processor: Key error... {0}'.format(e))
                # Proceed at all costs
                self.dropped_img.append(self.img_num)
            except Exception as e:
                logger.error('Processor error: {}: {} during image number {}'.format(type(e).__name__,
                                                                                            e, self.img_num))
                print(traceback.format_exc())
                self.dropped_img.append(self.img_num)
            self.total_times.append((time.time()-t)*1000.0)
        else:
            pass

# Should the following technically be internal?
    def _checkInput(self):
        ''' Check to see if we have images for processing
        From basic demo
        '''
        try:
            res = self.q_in.get()
            return res
        #TODO: additional error handling
        except Empty:
            pass
            # logger.info('No images for processing')
            # return None

    def _processImage(self, img):
        ''' Load data - here, .jpg image to tensor
        Input is already loaded image from q_in
        '''
        if img is None:
            raise ObjectNotFoundError
        else:
            # Takes np img (HWC) -> (NHWC) -> (NCHW)
            img = torch.from_numpy(img.copy())
            img = img.unsqueeze(dim=0).permute(0, 3, 1, 2)
        return img

    def _runInference(self, data):
        '''
        '''
        t = time.time()
        data = data.to(self.device)
        to_device = time.time() - t
        with torch.no_grad():
            # Would we want to squeeze for batch size > 1?
            output = self.model(data).to(self.device)
        return output, to_device
        
    def _classifyImage(self, output, labels=('plane', 'car', 'bird', 'cat', 'deer' 'dog', 'frog', 'horse', 'ship', 'truck')):
        '''
        '''
        _, predicted = np.argmax(output.data, dim=1)
        
        return labels[predicted]

class NaNDataException(Exception):
    pass