import time
import os
# import h5py
# import random
import numpy as np
import pandas as pd
from improv.store import CannotGetObjectError, ObjectNotFoundError
from queue import Empty

import torch
import torchaudio.transforms as TA
import torchvision.transforms as TV

from pathlib import Path
from queue import Empty

import traceback, warnings

from improv.actor import Actor, RunManager

import logging; logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class AudioProcessor(Actor):
    '''
    '''

    def __init__(self, *args, n_segs=None, model_path=None, params=None, gpu=None, gpu_num=None, time_opt=None, timing=None, out_path=None, method='fork', **kwargs):
        super().__init__(*args, **kwargs)
        self.done = False

        self.seg_num = 0

        if model_path is None:
            # logger.error("Must specify a model path.")
            logger.error("Must specify a model path.")
        else:
            self.model_path = model_path

        if gpu is True:
            self.device = torch.device("cuda:{}".format(gpu_num))
            torch.jit.fuser('fuser2')
        else:
            self.device = torch.device("cpu")
            torch.jit.fuser('fuser1')

        self.n_segs = n_segs

        self.params = params

        self.time_opt = time_opt
        if self.time_opt is True:
            self.timing = timing
            self.out_path = out_path

    def setup(self):
        os.makedirs(self.out_path, exist_ok=True)

        self.dropped_wav = []

        logger.info('Loading model for ' + self.name)

        t = time.time()        
        self.model = torch.jit.load(self.model_path).eval().to(self.device)
        torch.cuda.synchronize()
        load_model_time = (time.time() - t)*1000.0

        print('Time to load model:', load_model_time)
        with open(os.path.join(self.out_path, "load_model_time.txt"), "w") as text_file:
            text_file.write("%s" % load_model_time)
            text_file.close()

        t = time.time()
        sample_input = torch.rand(size=(1, 1, 128, 128), device=self.device)
        # self.model = torch.jit.optimize_for_inference(self.model)
        # Still have the high first run??? Why even though warmup in setup?
        with torch.no_grad():
            for _ in range(5):
                self.model(sample_input).to(self.device)
                torch.cuda.synchronize()
                
        warmup_time = (time.time() - t)*1000.0
        print('Time to warmup:', warmup_time)
        with open(os.path.join(self.out_path, "warmup_time.txt"), "w") as text_file:
            text_file.write("%s" % warmup_time)
            text_file.close()

    def run(self):
        '''
        '''
        self.proc_timestamps = []
        self.get_wav_out = []
        self.get_spec = []
        self.spec_to_np = []
        self.spec_to_store = []
        self.to_device = []
        self.inference_time = []
        self.z_to_np = []
        self.put_out_store = []
        self.put_q_out = []
        self.proc_total_times = []

        with RunManager(self.name, self.runProcessor, self.setup, self.q_sig, self.q_comm) as rm:
            print(rm)

        print('Processor broke, avg time per segment:', np.mean(self.proc_total_times))
        print('Processor got through', self.seg_num, ' segments')

        if self.time_opt is True:
            keys = self.timing
            values = [self.proc_timestamps, self.get_wav_out, self.get_spec, self.spec_to_np, self.spec_to_store, self.to_device, self.inference_time, self.z_to_np, self.put_out_store, self.put_q_out, self.proc_total_times]

        timing_dict = dict(zip(keys, values))
        df = pd.DataFrame.from_dict(timing_dict, orient='index').transpose()
        df.to_csv(os.path.join(self.out_path, 'proc_timing_' + str(self.n_segs) + '.csv'), index=False, header=True)
        
    def runProcessor(self):
        '''
        '''
        if self.done:
            pass

        self.proc_timestamps.append((time.time(), int(self.seg_num)))

        ids = self._checkInput()

        if ids is not None:
            t = time.time()
            self.done = False
            try:
                t1 = time.time()
                audio = self.client.getID(ids[0])
                fs = self.client.getID(ids[1])
                t2 = time.time()
                spec, spec_img = self._genSpec(audio, fs, self.params)
                t3 = time.time()
                spec = spec.detach().cpu().numpy()
                spec_img = spec_img.detach().cpu().numpy()
                t4 = time.time()
                spec_obj_id = self.client.put(spec, 'spec' + str(self.seg_num))
                spec_img_obj_id = self.client.put(spec_img, 'spec_img' + str(self.seg_num))
                t5 = time.time()                
                latents, t_dev, t_inf = self._runInference(spec_img)
                t6 = time.time()
                latents = latents.detach().cpu().numpy()
                t7 = time.time()
                latents_obj_id = self.client.put(latents, 'latents' + str(self.seg_num))
                t8 = time.time()
                # self.q_out.put([spec_obj_id, spec_img_obj_id, latents_obj_id, str(self.seg_num)])
                self.put_q_out.append((time.time() - t8)*1000.0)

                self.get_wav_out.append((t2 - t1)*1000.0)
                self.get_spec.append((t3 - t2)*1000.0)
                self.spec_to_np.append((t4 - t3)*1000.0)
                self.spec_to_store.append((t5 - t4)*1000.0)
                self.to_device.append(t_dev*1000.0)
                self.inference_time.append(t_inf*1000.0)
                self.z_to_np.append((t7 - t6)*1000.0)
                self.put_out_store.append((t8 - t7)*1000.0)

                self.seg_num += 1
                self.proc_total_times.append((time.time() - t)*1000.0)

            # Insert exceptions here...ERROR HANDLING, SEE ANNE'S ACTORS - from 1p demo
            except ObjectNotFoundError:
                logger.error('Processor: Image {} unavailable from store, dropping'.format(self.seg_num))
                self.dropped_wav.append(self.seg_num)
                # self.q_out.put([1])
            except KeyError as e:
                logger.error('Processor: Key error... {0}'.format(e))
                # Proceed at all costs
                self.dropped_wav.append(self.seg_num)
            except Exception as e:
                logger.error('Processor error: {}: {} during image number {}'.format(type(e).__name__,
                                                                                            e, self.seg_num))
                print(traceback.format_exc())
                self.dropped_wav.append(self.seg_num)
            self.proc_total_times.append((time.time() - t)*1000.0)
        else:
            pass

        if self.seg_num == self.n_segs:
            logger.error('Done processing all available data: {}'.format(self.seg_num))
            self.data = None
            self.q_comm.put(None)
            self.done = True  # stay awake in case we get a shutdown signal

    def _checkInput(self):
        ''' Check to see if we have .wav — q_in
        '''
        try:
            res = self.q_in.get(timeout=0.005)
            return res
        #TODO: additional error handling
        except Empty:
            pass
            # logger.info('No .wav files for processing')
            # return None

    def _genSpec(self, audio, fs, params):
        '''
        '''
        #NOT WRITABLE ERROR:
        audio_t = torch.Tensor(audio).to(self.device)

        n_fft = params['nperseg']

        # transforms = torch.nn.Sequential(
        #     TA.Spectrogram(
        #         n_fft=n_fft,
        #         win_length=params['nperseg'],
        #         hop_length=params['noverlap'],
        #         center=True,
        #         power=2.0),
        #     TA.MelScale(
        #         sample_rate=fs,
        #         f_min = params['f_min'], f_max = params['f_max'], 
        #         n_fft=n_fft)).to(self.device)

        transforms = TA.Spectrogram(
            n_fft=n_fft,
            win_length=params['win_len'],
            hop_length=params['noverlap'],
            center=True,
            power=2.0).to(self.device)

        spec = transforms(audio_t).to(self.device)
            
        transform = TV.Resize((params['height'], params['width'])).to(self.device)

        spec_img = transform(spec.unsqueeze(0))

        return spec, spec_img

    def _runInference(self, spec_img):
        '''
        '''
        t = time.time()
        spec = torch.Tensor(spec_img).to(self.device)
        torch.cuda.synchronize()        
        to_device = time.time() - t
        with torch.no_grad():
            t = time.time()
            output = self.model.forward(spec.unsqueeze(0))
            torch.cuda.synchronize()
            inf_time = time.time() - t

        return output, to_device, inf_time