import numpy as np
import os
import pandas as pd
from queue import Empty
import time

from improv.store import CannotGetObjectError, ObjectNotFoundError
from improv.actor import Actor

import torch
from scipy.special import softmax # why scipy? — refactor for use torch Softmax, torch.nn.Softmax(dim=None) or torch.softmax

import logging; logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import traceback

class CNNProcessor(Actor):
    # TODO: Update ALL docstrings
    # TODO: Clean commented sections
    # TODO: Add any relevant q_comm
    ''' 
    '''

    def __init__(self, *args, n_imgs=None, gpu=False, gpu_num=None, model_path=None, classify=False, labels=None, time_opt=False, timing=None, lab_timing=None, out_path=None, method='spawn', **kwargs):
        super().__init__(*args, **kwargs)
        logger.info(model_path)
        if model_path is None:
            logger.error("Must specify a model path.")
        else:
            self.model_path = model_path

        self.img_num = 0

        if gpu is True:
            self.device = torch.device("cuda:{}".format(gpu_num))
            torch.jit.fuser('fuser2')
        else:
            self.device = torch.device("cpu")
            torch.jit.fuser('fuser1')

        self.classify = classify
        self.labels = labels

        self.n_imgs = n_imgs

        self.time_opt = time_opt
        if self.time_opt is True:
            self.timing = timing
            self.lab_timing = lab_timing
            self.out_path = out_path

    def setup(self):
        ''' Initialize model
        '''
        os.makedirs(self.out_path, exist_ok=True)

        self.proc_timestamps = []
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
            self.true_label = []
            self.pred_label = []
            self.percent = []
            self.top_five = []

        self.proc_total_times = []

        logger.info('Loading model for ' + self.name)
        self.done = False
        self.dropped_img = []

        t = time.time()        
        self.model = torch.jit.load(self.model_path).eval().to(self.device)
        torch.cuda.synchronize()
        load_model_time = (time.time() - t)*1000.0

        print('Time to load model:', load_model_time)
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
        print('Time to warmup:', warmup_time)
        with open(self.out_path + "warmup_time.txt", "w") as text_file:
            text_file.write("%s" % warmup_time)
            text_file.close()

        with open(self.labels, "r") as f:
            self.labels = f.read().split(", ")
            f.close()

    def run(self):
        ''' 
        Run processor continually on input data, e.g.,images
        
        Run process. Runs once per image.
            Output is a location in the DS to continually
            place the processed image, model output, and classification/prediction with ref number that
            corresponds to the frame number (TODO)
            [Adapted from neurofinder/actors/processor.py]
        '''
        
        self.proc_timestamps.append((time.time(), int(self.img_num)))

        ids = self._checkInput()

        if ids is not None:
            t = time.time()
            self.done = False
            try:
                t1 = time.time()
                img = self.client.getID(ids[0])
                t2 = time.time()
                img = self._processImage(img)
                t3 = time.time()
                output, t_dev, t_inf = self._runInference(img)
                t4 = time.time()
                # Want to avoid copy to CPU to detach...detach first?
                features = output[0].detach().cpu().numpy()
                predictions = output[1].detach().cpu().numpy()
                t5 = time.time()
                feat_obj_id = self.client.put(features, 'features' + str(self.img_num))
                pred_obj_id = self.client.put(predictions, 'predictions' + str(self.img_num))
                t6 = time.time()
                if self.classify is True and self.labels is not None:
                # Do number correct in visual? Output w/pred?
                # Compare label in and label out, add to counter if correct, keep note of number for each class...compare list as both fill in, frame-by-frame? Updating confusion matrix?
                    pred_label, percent, top_five = self._classifyImage(predictions, self.labels)
                    t7 = time.time()
                    lab_obj_id = self.client.put(pred_label, 'prediction' + str(self.img_num))
                    percent_obj_id = self.client.put(percent, 'percent' + str(self.img_num))
                    top_obj_id = self.client.put(top_five, 'top_five' + str(self.img_num))
                    t8 = time.time()
                    self.true_label.append(self.labels[self.client.getID(ids[1])])
                    self.pred_label.append(pred_label)
                    self.percent.append(percent)
                    self.top_five.append(top_five)
                    # self.q_out.put([feat_obj_id, pred_obj_id, lab_obj_id, percent_obj_id, top_obj_id, str(self.img_num)])
                    self.put_q_out.append((time.time() - t8)*1000.0)
                    self.pred_time.append((t7 - t6)*1000.0)
                    self.put_pred_store.append((t8 - t7)*1000.0)
                    # Visualize updated accuracy...update every few images? OR every image?
                    # Confusion matrix that updates w/incoming data
                else:
                    # self.q_out.put([out_obj_id, str(self.img_num)])
                    self.put_q_out.append((time.time() - t5)*1000.0)

                self.get_img_out.append((t2 - t1)*1000.0)
                self.proc_img_time.append((t3 - t2)*1000.0)
                self.to_device.append(t_dev*1000.0)
                self.inference_time.append(t_inf*1000.0)
                self.out_to_np.append((t5 - t4)*1000.0)
                self.put_out_store.append((t6 - t5)*1000.0)

                self.img_num += 1
                self.proc_total_times.append((time.time() - t)*1000.0)

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
            self.proc_total_times.append((time.time() - t)*1000.0)
        else:
            pass

        # From bubblewrap demo and FileAcquirer
        if self.img_num == self.n_imgs:
            logger.error('Done processing all available data: {}'.format(self.img_num))
            self.data = None
            self.q_comm.put(None)
            self.done = True  # stay awake in case we get a shutdown signal
        print('Processor broke, avg time per image:', np.mean(self.proc_total_times))
        print('Processor got through', self.img_num, ' images')

        # keys = ['proc_timestamps', 'get_img_out', 'proc_img_time', 'to_device', 'inference_time', 'out_to_np', 'put_out_store', 'put_q_out', 'total_times']

    def stop(self):
        '''
        '''

        keys = self.timing
        values = [self.proc_timestamps, self.get_img_out, self.proc_img_time, self.to_device, self.inference_time, self.out_to_np, self.put_out_store, self.put_q_out, self.proc_total_times]

        if self.classify is True and self.labels is not None:
            self.lab_timing.extend(['true_label', 'pred_label', 'percent', 'top_five'])
            keys.extend(self.lab_timing)
            values.extend([self.pred_time, self.put_pred_store, self.true_label, self.pred_label, self.percent, self.top_five])

        timing_dict = dict(zip(keys, values))
        df = pd.DataFrame.from_dict(timing_dict, orient='index').transpose()
        df.to_csv(os.path.join(self.out_path, 'proc_timing_' + str(self.n_imgs) + '.csv'), index=False, header=True)


# Should the following technically be internal?
    def _checkInput(self):
        ''' Check to see if we have images for processing
        From basic demo
        '''
        try:
            res = self.q_in.get(timeout=0.005)
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
            img = torch.as_tensor(img.copy())
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
            t = time.time()
            # Would we want to squeeze for batch size > 1?
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