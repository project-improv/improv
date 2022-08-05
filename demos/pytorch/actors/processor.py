import os
import time
import numpy as np
from improv.store import CannotGetObjectError, ObjectNotFoundError
from queue import Empty
from improv.actor import Actor, RunManager

import torch
import torch.multiprocessing as mp
from torchvision.io import read_image

# mp.set_start_method('spawn')

# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

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

    def __init__(self, *args, gpu=False, gpu_num=None, model_path=None, classify=False, **kwargs):
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

    def setup(self):
        ''' Initialize model
        '''
        # Idk where this should be?
        os.makedirs("output", exist_ok=True)
        logger.info('Loading model for ' + self.name)
        self.done = False
        self.dropped_img = []

        # try:
        #     mp.set_start_method("forkserver")
        # except: pass

        t = time.time()

        self.model = torch.jit.load(self.model_path).to(self.device)

        # self.model = torch.jit.load(self.model_path).to(self.device).share_memory

        # num_processes = 4
        # processes = []
        # for rank in range(num_processes):
        #     p = mp.Process(target=self.transforms, args=(self.model,))
        #     p.start()
        #     processes.append(p)
        # for p in processes:
        #     p.join()

        load_model_time = time.time() - t
        print('Time to load model: ', load_model_time*1000.0)
        with open("output/timing/load_model_time.txt", "w") as text_file:
            text_file.write("%s" % load_model_time)

    def run(self):
        ''' Run the processor continually on input data, e.g.,images
        '''
        self.get_img_out = []
        self.proc_img_time = []
        self.to_device = []
        self.inference_time = []
        self.out_to_np = []
        self.put_out_store = []
        self.put_q_out = []
        if self.classify is True:
            self.pred_time = []
            self.put_pred_store = []
            self.put_pred_out = []

        self.total_times = []
        
        with RunManager(self.name, self.runProcess, self.setup, self.q_sig, self.q_comm) as rm:
            print(rm)
        
        print('Processor broke, avg time per image: ', np.mean(self.total_times, axis=0))
        print('Processor got through ', self.img_num, ' images')

        np.savetxt('output/timing/get_img_out.txt', np.array(self.get_img_out))
        np.savetxt('output/timing/process_image_time.txt', np.array(self.proc_img_time))
        np.savetxt('output/timing/to_device.txt', np.array(self.to_device))
        np.savetxt('output/timing/inference_time.txt', np.array(self.inference_time))
        np.savetxt('output/timing/out_to_np.txt', np.array(self.out_to_np))
        np.savetxt('output/timing/put_out_store.txt', np.array(self.put_out_store))
        np.savetxt('output/timing/put_q_out.txt', np.array(self.put_q_out))
        if self.classify is True:
            np.savetxt('output/timing/prediction_time.txt', np.array(self.pred_time))
            np.savetxt('output/timing/put_prediction_time.txt', np.array(self.put_pred_store))
            np.savetxt('output/timing/put_prediction_time.txt', np.array(self.put_pred_out))
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
                output, t_dev = self._runInference(img)
                t3 = time.time()
                output = output.detach().numpy()
                t4 = time.time()
                out_obj_id = self.client.put(output, 'output' + str(self.img_num))
                t5 = time.time()
                self.q_out.put([[out_obj_id, str(self.img_num)]])
                t6 = time.time()
                if self.classify is True:
                    pred = self._classifyImage(output, self.img_num)
                    t7 = time.time()
                    pred_obj_id = self.client.put(pred, 'prediction' + str(self.img_num))
                    t8 = time.time()
                    self.q_out.put([[pred_obj_id, str(self.img_num)]])
                    t9 = time.time()
                    self.pred_time.append((t7 - t6)*1000.0)
                    self.put_pred_store.append((t8 - t7)*1000.0)
                    self.put_pred_out.append((t9 - t8)*1000.0)


                self.get_img_out.append((t1 - t)*1000.0)
                self.proc_img_time.append((t2 - t1)*1000.0)
                self.to_device.append(t_dev*1000.0)
                self.inference_time.append((t3 - t2)*1000.0)
                self.out_to_np.append((t4 - t3)*1000.0)
                self.put_out_store.append((t5 - t4)*1000.0)
                self.put_q_out.append((t6 - t5)*1000.0)

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
                                                                                            e, self.img_num))
                print(traceback.format_exc())
                self.dropped_img.append(self.img_num)
            self.total_times.append((time.time()-t)*1000.0)
        else:
            pass

# Should the following technically be internal?
    def _checkImages(self):
        ''' Check to see if we have images for processing
        From basic demo
        '''
        try:
            # Timeout = 0 ms
            res = self.q_in.get(timeout=0)
            return res
        #TODO: additional error handling
        except Empty:
            logger.info('No images for processing')
            return None

    def _processImage(self, img):
        ''' Load data - here, .jpg image to tensor
        Input is already loaded image from q_in
        '''
        if img is None:
            raise ObjectNotFoundError
        # img = torch.from_numpy(img).unsqueeze(dim=0)
        img = read_image(img)
        return img

    def _runInference(self, data):
        '''
        '''
        # TODO: time to device
        t = time.time()
        data = data.unsqueeze(dim=0).to(self.device)
        t_dev = time.time() - t
        with torch.no_grad():
            output = self.model(data).squeeze(dim=0).to(self.device)
        return output, t_dev

    # def _classifyImage(self, labels=('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')):
    #     '''
    #     '''
    #     _, predicted = torch.max(self.output.data, 1)

    #     with open(self.label_path + "label/{}.txt".format(self.img_num), "r") as text_file:
    #         text_file.read("%s" % self.load_transforms_time)

    #     correct = (predicted == labels).sum().item()

class NaNDataException(Exception):
    pass