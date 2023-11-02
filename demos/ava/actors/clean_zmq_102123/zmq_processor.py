import os
from queue import Empty
import time
import traceback
import warnings

import numpy as np
import pandas as pd
import torch
import yaml

from ava.models.vae import VAE
from ava.preprocessing.utils import get_spec

from improv.actor import Actor
from improv.store import CannotGetObjectError, ObjectNotFoundError

import logging; logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ZMQAudioProcessor(Actor):
    """Actor for processing audio acquired via ZMQ. Receives audio from ZMQAudioAcquirer (audio q_in), computes a spectrogram, and produces VAE output, latent means.
    """

    def __init__(self, *args, model_path=None, params=None, gpu=None, gpu_num=None, time_opt=None, timing=None, out_path=None, method='fork', **kwargs):
        super().__init__(*args, **kwargs)
        
        self.done = False

        if model_path is None:
            logger.error("Must specify a model path.")
        else:
            self.model_path = model_path
            logger.info(f"Model path: {self.model_path}")

        if gpu:
            if gpu_num is not None:
                self.device = torch.device("cuda:{}".format(gpu_num))
            else: 
                self.device = torch.device("cuda")
            logger.info(f"Device: {self.device}")
            torch.jit.fuser('fuser2')
        else:
            self.device = torch.device("cpu")
            logger.info(f"Device: {self.device}")
            torch.jit.fuser('fuser1')

        self.params = params

        self.out_path = out_path

        self.time_opt = time_opt
        if self.time_opt:
            self.timing = timing
            self.timing_path = os.path.join(out_path, 'timing')
            logger.info(f"{self.name} timing directory: {self.timing_path}")

    def __str__(self):
        return f"Name: {self.name}, Data: {self.data}"
    
    def setup(self):
        """Setup for AVA ZMQAudioProcessor.
        """
        logger.info(f"Running setup for {self.name}.")

        self.proc_total_times = []
        self.seg_nums = []

        # device = "cuda" if torch.cuda.is_available() else "cpu"

        if self.time_opt:
            os.makedirs(self.timing_path, exist_ok=True)
            logger.info(f"Initializing lists for {self.name} timing.")
            self.proc_timestamps = []
            self.get_spec = []
            self.inference_time = []
            self.t_recv = []

        with open(self.params, 'r') as file:
            self.params = yaml.safe_load(file)
            logger.info(f"params: {self.params}")

        logger.info(f"{self.name} loading model: {self.model_path}")
        t = time.perf_counter_ns()
        self.model = VAE().eval().to(self.device)
        self.model = torch.jit.load(self.model_path).eval().to(self.device)
        torch.cuda.synchronize()
        load_model_time = (time.perf_counter_ns() - t) * 10**-6
        logger.info(f"Time to load model: {load_model_time} ms")
        if self.time_opt:
            with open(os.path.join(self.timing_path, "load_model_time.txt"), "w") as text_file:
                text_file.write("%s" % load_model_time)
                text_file.close()
                
        logger.info(f"{self.name} warming up model.")
        t = time.perf_counter_ns()
        sample_input = torch.rand(size=(1,128,128), device=self.device)
        with torch.no_grad():
            for _ in range(100):
                self.model(sample_input)
                torch.cuda.synchronize()
        warmup_time = (time.perf_counter_ns() - t) * 10**-6
        logger.info(f"Time to warmup model: {warmup_time} ms")
        if self.time_opt:
            with open(os.path.join(self.timing_path, "warmup_time.txt"), "w") as text_file:
                text_file.write("%s" % warmup_time)
                text_file.close()

        self.seg_num = 0

        logger.info(f"Completed setup for {self.name}.")

    def stop(self):
        """Stop procedure — save out timing information.
        """
        logger.info(f"{self.name} stopping.")

        logger.info(f"Processor avg time per segment: {np.mean(self.proc_total_times)}")
        logger.info(f"Processor got through {self.seg_num} segments.")

        if self.time_opt:
            logger.info(f"Saving timing info for {self.name}.")
            keys = self.timing
            values = [self.proc_timestamps, self.t_recv, self.get_spec, self.inference_time, self.proc_total_times, self.seg_nums]

            timing_dict = dict(zip(keys, values))
            df = pd.DataFrame.from_dict(timing_dict, orient='index').transpose()
            df.to_csv(os.path.join(self.timing_path, 'proc_timing_' + str(self.seg_num) + '.csv'), index=False, header=True)

        logger.info(f"{self.name} stopped.")

        return 0
        
    def runStep(self):
        """Run step — receive audio from acquirer, compute spectrogram, run inference (compute latent means), send out VAE output (latent means), q_out.
        """
        if self.done:
            pass
        
        t = time.perf_counter_ns()

        if self.time_opt:
            self.proc_timestamps.append(t)

        ids = self._checkInput()

        if ids is not None:
            t = time.perf_counter_ns()
            self.done = False
            try:
                audio = self.client.getID(ids[0])
                daq_timestamp = self.client.getID(ids[1])

                t_recv = time.perf_counter_ns()
                spec = self._genSpec(0, self.params['win_len'], audio, self.params['fs'], self.params)
                t_spec = time.perf_counter_ns()

                latents, t_inf = self._runInference(spec)
                # latents = latents[0].detach().cpu().numpy()
                # latents_obj_id = self.client.put(latents, 'latents' + str(self.seg_num))

                # self.q_out.put([latents_obj_id, str(self.seg_num)])
                timestamp_obj_id = self.client.put(daq_timestamp, 'daq_timestamp' + str(self.seg_num))
                self.q_out.put([timestamp_obj_id, str(self.seg_num)])

                if self.time_opt:
                    self.seg_nums.append(int(self.seg_num))
                    self.t_recv.append(t_recv)
                    self.get_spec.append(t_spec)
                    self.inference_time.append(t_inf)

            except ObjectNotFoundError:
                logger.error(f"{self.name}: Audio {self.seg_num} unavailable from store, dropping")
                # self.dropped_wav.append(self.seg_num)
            except KeyError as e:
                logger.error(f"{self.name}: Key error... {e}")
                # self.dropped_wav.append(self.seg_num)
            except Exception as e:
                logger.error(f"{self.name} error: {type(e).__name__}: {e} during segment number {self.seg_num}")
                logger.info(traceback.format_exc())
                # self.dropped_wav.append(self.seg_num)

        else:
            pass

        # if ids is None:
        #     if self.time_opt:
        #         self.seg_nums.append(self.seg_num)
        #         self.get_spec.append(np.nan)
        #         self.inference_time.append(np.nan)
            
        self.seg_num += 1
        self.proc_total_times.append((time.perf_counter_ns() - t) * 10**-6)

        self.data = None
        self.q_comm.put(None)
        self.done = True  # stay awake in case we get a shutdown signal

    def _checkInput(self):
        """Check to see if we have audio, q_in.

        Returns:
            _type_: _description_
        """
        try:
            res = self.q_in.get(timeout=0.005)
            return res
        #TODO: additional error handling
        except Empty:
            # logger.info('No audio for processing')
            pass

    def _genSpec(self, onset, offset, audio, fs, params):
        """Compute spectrogram.

        Args:
            onset (_type_): _description_
            offset (_type_): _description_
            audio (_type_): _description_
            fs (_type_): _description_
            params (_type_): _description_
            target_times (_type_, optional): _description_. Defaults to None.

        Returns:
            spec: _description_
        """
        spec, _ = get_spec(onset, offset, audio, params, fs)

        return spec

    def _runInference(self, spec):
        """Run inference — input audio to VAE, run inference, output latent means.

        Args:
            spec (_type_): _description_

        Returns:
            output: _description_
            t_inf: _description_
        """
        spec = torch.Tensor(spec).to(self.device)
        with torch.no_grad():
            output = self.model(spec.unsqueeze(0))
        torch.cuda.synchronize()
        t_inf = time.perf_counter_ns()

        return output, t_inf