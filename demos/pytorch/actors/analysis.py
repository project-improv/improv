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

class PyTorchAnalysis(Actor):
    '''
    '''

    def __init__(self, *args, model_path=None, label_path=None, classify=True,):
        super().__init__(*args)
        logger.info(model_path)
        if model_path is None:
            # logger.error("Must specify a pre-trained model path.")
            logger.error("Must specify a JIT-compiled pre-trained model path.")
        else:
            self.model_path = model_path
        if classify is True and label_path is None:
            logger.error("Must specify a path to image labels.")
        else:
            self.label_path = label_path
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def setup(self):
        ''' Initialize pre-trained model
        '''
        logger.info('Loading model for ' + self.name)
        self.done = False

        t = time.time()

        self.model= torch.jit.load(self.model_path).to(self.device)
        
        self.load_model_time = time.time() - t
        print('Time to load model: ', self.load_model_time*1000.0)
        with open("output/timing/load_model_time.txt", "w") as text_file:
            text_file.write("%s" % self.load_model_time)

        # return self.model

    def run(self):
        '''
        '''
        self.img_num = []
        self.to_gpu = []
        self.inference_time = []
        self.put_out_time = []
        self.pred_time = []
        self.put_pred_time = []
        self.total_times = []

        with RunManager(self.name, self.runAnalysis, self.setup, self.q_sig, self.q_comm) as rm:
            print(rm)
        
        print('Analysis broke, avg time per image: ', np.mean(self.total_times, axis=0))
        print('Analysis got through ', self.img_num, ' images')

        np.savetxt('output/timing/to_gpu.txt', np.array(self.to_gpu))
        np.savetxt('output/timing/inference_time.txt', np.array(self.inference_time))
        np.savetxt('output/timing/put_out_time.txt', np.array(self.put_out_time))
        if self.classify is True:
            np.savetxt('output/timing/prediction_time.txt', np.array(self.pred_time))
            np.savetxt('output/timing/put_prediction_time.txt', np.array(self.put_pred_time))
        np.savetxt('output/timing/total_times.txt', np.array(self.total_times))

    def runAnalysis(self):
        ''' Run analysis, i.e., inference. 
        Runs once per image. 
        Output is a location in the DS to continually place the model output and classification/prediction, with ref number that corresponds to the frame number (TODO)
        '''
        ids = self._checkProcImages()

        if ids is not None:
            t = time.time()
            self.done = False
            try:
                img = self.client.getID(ids[0])
                t1 = time.time()
                img.to(self.device)
                t2 = time.time()
                output = self._runInference(img)
                t3 = time.time()
                out_obj_id = self.client.put(output.numpy(), 'output' + str(self.img_num))
                self.q_out.put([out_obj_id, str(self.img_num)])
                t4 = time.time()
                if self.classify is True:
                    pred = self._classifyImage(output, self.img_num)
                    t5 = time.time()
                    pred_obj_id = self.client.put(pred, 'prediction' + str(self.img_num))
                    # Check if can put out twice?
                    self.q_out.put([[pred_obj_id, str(self.img_num)]])
                    t6 = time.time()
                    self.pred_time.append((t5 - t4)*1000.0)
                    self.put_pred_time.append((t6 - t5)*1000.0)
                self.to_gpu.append((t2 - t1)*1000.0)
                self.inference_time.append((t3 - t2)*1000.0)
                self.put_out_time.append((t4 - t5)*1000.0)
                
                self.img_num += 1

            # Insert exceptions here...ERROR HANDLING, SEE ANNE'S ACTORS - from 1p demo
            except ObjectNotFoundError:
                logger.error('Analysis: Image {} unavailable from store, dropping'.format(self.img_num))
                self.dropped_img.append(self.img_num)
                self.q_out.put([1])
            except KeyError as e:
                logger.error('Analysis: Key error... {0}'.format(e))
                # Proceed at all costs
                self.dropped_img.append(self.img_num)
            except Exception as e:
                logger.error('Analysis error: {}: {} during image number {}'.format(type(e).__name__,
                                                                                            e, self.frame_number))
                print(traceback.format_exc())
                self.dropped_img.append(self.img_num)
            self.total_times.append(time.time()-t)
        else:
            pass

    def _checkProcImages(self):
        ''' Check to see if we have processed images for analysis (should be tensors (1, 3, 224, 224)) - check instance and size? TODO?
        '''
        # t = time.time()
        try:
            # Timeout = 0 ms
            res = self.q_in.get(timeout=0)
            return res
        #TODO: additional error handling
        except Empty:
            logger.info('No processed images for analysis')
            return None
        # self.get_in_time = time.time() - t

    def _runInference(self, data):
        '''
        '''
        with torch.no_grad():
            output = self.model(data).squeeze(dim=0).to(self.device)
        return output

# def _classifyImage(self, classes=('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')):
#     '''
#     '''
#     _, predicted = torch.max(self.output.data, 1)

#         total = labels.size(0)
#         correct = (predicted == labels).sum().item()