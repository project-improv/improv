import os
import json
import time
import numpy as np
from improv.store import CannotGetObjectError, ObjectNotFoundError
from queue import Empty
from improv.actor import Actor, RunManager

import torch
from scipy.special import softmax

import logging; logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.getLogger('PIL').setLevel(logging.WARNING)

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
            # self.device = torch.device("cuda:{}".format(gpu_num) if torch.cuda.is_available() else "cpu")
            self.device = torch.device("cuda:{}".format(gpu_num))
            torch.jit.fuser('fuser2')
        else:
            self.device = torch.device("cpu")
            torch.jit.fuser('fuser1')

        self.classify = classify

        if self.classify is True and labels is None:
            logger.error("Must specify labels for classification.")
        elif self.classify is True and labels is not None:
            self.labels = labels

        self.out_path = out_path

    def setup(self):
        ''' Initialize model
        '''
        if self.classify is True and self.labels is not None:
            logger.info('Loading label names')
            with open(self.labels, "r") as label_file:
                self.labels = json.loads(label_file.read())
                label_file.close()

        # Idk where this should be?
        os.makedirs(self.out_path, exist_ok=True)
        logger.info('Loading model for ' + self.name)
        self.done = False
        self.dropped_img = []

        t = time.time()        
        self.model = torch.jit.load(self.model_path).eval().to(self.device)
        torch.cuda.synchronize()
        load_model_time = (time.time() - t)*1000.0

        # self.model = torch.jit.load(self.model_path).to(self.device).share_memory

        print('Time to load model: ', load_model_time)
        with open(self.out_path + "load_model_time.txt", "w") as text_file:
            text_file.write("%s" % load_model_time)
            text_file.close()

        t = time.time()
        sample_input = torch.rand(size=(1, 3, 224, 224), device=self.device)
        # self.model = torch.jit.optimize_for_inference(self.model)
        # Still have the high first run??? Why even though warmup in setup?
        with torch.no_grad():
            for _ in range(5):
                self.model(sample_input)
                torch.cuda.synchronize()
                
        warmup_time = (time.time() - t)*1000.0
        print('Time to warmup: ', warmup_time)
        with open(self.out_path + "warmup_time.txt", "w") as text_file:
            text_file.write("%s" % warmup_time)
            text_file.close()

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
            self.true_label = []
            self.pred_label = []
            self.percent = []
            self.top_five = []

        self.total_times = []

        # print('TEST!!!')

        with RunManager(self.name, self.runProcess, self.setup, self.q_sig, self.q_comm, runStore=self._getStoreInterface()) as rm:
            print(rm)

        # logger.error('1: TEST!!!TEST!!!TEST!!!TEST!!!')

        if self.img_num == 300:
            with open(self.out_path + "proc_vars_rm.txt", "w") as f:
                print(vars(self), file=f)
                f.close()

        # logger.error('2: TEST!!!TEST!!!TEST!!!TEST!!!')
        
        print('Processor broke, avg time per image: ', np.mean(self.total_times, axis=0))
        print('Processor got through ', self.img_num, ' images')

        # np.savetxt(self.out_path + 'get_img_out.txt', np.array(self.get_img_out))

        # logger.error('3: TEST!!!TEST!!!TEST!!!TEST!!!')

        # np.savetxt(self.out_path + 'process_img_time.txt', np.array(self.proc_img_time))
        # np.savetxt(self.out_path + 'to_device.txt', np.array(self.to_device))
        # np.savetxt(self.out_path + 'inference_time.txt', np.array(self.inference_time))
        # np.savetxt(self.out_path + 'out_to_np.txt', np.array(self.out_to_np))
        # np.savetxt(self.out_path + 'put_out_store.txt', np.array(self.put_out_store))
        # np.savetxt(self.out_path + 'put_q_out.txt', np.array(self.put_q_out))
        # if self.classify is True:
        #     np.savetxt(self.out_path + 'prediction_time.txt', np.array(self.pred_time))
        #     np.savetxt(self.out_path + 'put_prediction_store.txt', np.array(self.put_pred_store))
        #     np.savetxt(self.out_path + 'put_prediction_dict.txt', np.array(self.put_pred_out))
        #     logger.error('4: TEST!!!TEST!!!TEST!!!TEST!!!')
        # np.savetxt(self.out_path + 'put_prediction_out.txt', np.array(self.put_q_out))
        # np.savetxt(self.out_path + 'total_times.txt', np.array(self.total_times))

        # print('Done saving out!')

    def runProcess(self):
        ''' Run process. Runs once per image.
            Output is a location in the DS to continually
            place the processed image, model output, and classification/prediction with ref number that
            corresponds to the frame number (TODO)
            [From neurofinder/actors/processor.py]
        '''
        ids = self._checkInput()

        if ids is not None:
            if self.classify is True:
                n_imgs = ids[3]
            else:
                n_imgs = ids[2]
            t = time.time()
            self.done = False
            try:
                # Maybe switch ids 0 and 1?
                img = self.client.getID(ids[0])
                t1 = time.time()
                img = self._processImage(img)
                output, t_dev, t_inf = self._runInference(img)
                t2 = time.time()
                # Want to avoid copy to CPU to detach...detach first?
                features = output[0].to('cpu').numpy()
                predictions = output[1].to('cpu').numpy()
                t3 = time.time()
                feat_obj_id = self.client.put(features, 'features' + str(self.img_num))
                pred_obj_id = self.client.put(predictions, 'predictions' + str(self.img_num))
                t4 = time.time()
                if self.classify is True:
                # Do number correct in visual? Or output w/pred?
                # Compare label in and label out, add to counter if correct, keep note of number for each class...but can compare list as both fill in, frame-by-frame? Updating confusion matrix?
                    pred_label, percent, top_five = self._classifyImage(predictions, self.labels)
                    t5 = time.time()
                    lab_obj_id = self.client.put(pred_label, 'prediction' + str(self.img_num))
                    percent_obj_id = self.client.put(percent, 'percent' + str(self.img_num))
                    top_obj_id = self.client.put(top_five, 'top_five' + str(self.img_num))
                    t6 = time.time()
                    self.true_label.append(self.client.getID(ids[1]))
                    self.pred_label.append(pred_label)
                    self.percent.append(percent)
                    self.top_five.append(top_five)
                    # AttributeError: 'NoneType' object has no attribute 'put'
                    # self.q_out.put([feat_obj_id, pred_obj_id, lab_obj_id, percent_obj_id, top_obj_id, str(self.img_num)])
                    self.put_q_out.append((time.time() - t6)*1000.0)
                    self.pred_time.append((t5 - t4)*1000.0)
                    self.put_pred_store.append((t6 - t5)*1000.0)
                    # Visualize updated accuracy...update every few images? OR every image?
                    # Confusion matrix that updates w/incoming data
                else:
                    # self.q_out.put([out_obj_id, str(self.img_num)])
                    self.put_q_out.append((time.time() - t4)*1000.0)

                self.get_img_out.append((t1 - t)*1000.0)
                self.proc_img_time.append((t2 - t1)*1000.0)
                self.to_device.append(t_dev*1000.0)
                self.inference_time.append(t_inf*1000.0)
                self.out_to_np.append((t3 - t2)*1000.0)
                self.put_out_store.append((t4 - t3)*1000.0)

                self.img_num += 1
                self.total_times.append((time.time() - t)*1000.0)

                if self.img_num == n_imgs:
                #    self.pred_out = {'true_label': self.true_label, 'pred_label': self.pred_label, 'percent': self.percent, 'top_five': self.top_five}
                    tmp = vars(self)
                    with open(self.out_path + "proc_vars.txt", "w+") as f:    
                        d = {str(item): str(tmp[item]) for item in tmp}
                        print(d, file=f)
                        # print(vars(self), file=f)
                        f.close()
                        # print('Done saving!')

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
            self.total_times.append((time.time() - t)*1000.0)

            # if self.img_num == len(self.files):
            #     with open(self.out_path + "proc_vars.txt", "w") as f:
            #         print(vars(self), file=f)
            #         f.close()
        else:
            pass

        # From bubblewrap demo and FileAcquirer
        if self.img_num == n_imgs:
            logger.error('Done processing all available data: {}'.format(self.img_num))
            self.data = None
            self.q_comm.put(None)
            self.done = True  # stay awake in case we get a shutdown signal


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
        torch.cuda.synchronize()        
        to_device = time.time() - t
        with torch.no_grad():
            # Would we want to squeeze for batch size > 1?
            t = time.time()
            output = self.model(data)
            torch.cuda.synchronize()
            inf_time = time.time() - t

        return output, to_device, inf_time
        
    def _classifyImage(self, predictions, labels):
        '''
        output = features (1, 1000), predictions(1, 10)
        '''
        index = np.argmax(predictions, axis=1)[0]  
        percentage = (softmax(predictions) * 100)[0]

        indices = np.argsort(predictions, axis=1)[::-1][0]
        top_five = [(labels[idx], percentage[idx].item()) for idx in indices[:5]]
        
        return labels[index], percentage[index], top_five

class NaNDataException(Exception):
    pass