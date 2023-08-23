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

from ava.models.vae import VAE
from ava.preprocessing.utils import get_spec as get_interp_spec
from ava.segmenting.utils import get_spec
from ava.plotting.shotgun_movie import SimpleDataset

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
            logger.info(f"Model path: {self.model_path}")

        if gpu:
            self.device = torch.device("cuda:{}".format(gpu_num))
            torch.jit.fuser('fuser2')
        else:
            self.device = torch.device("cpu")
            torch.jit.fuser('fuser1')

        self.n_segs = n_segs

        self.save_stft = params['save_stft']
        self.save_spec = params['save_spec']
        self.save_latent = params['save_latent']
        
        self.win_len = params['win_len']

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
        """Setup for AVA AudioProcessor.
        """

        logger.info(f"Running setup for {self.name}.")

        if self.time_opt:
            os.makedirs(self.timing_path, exist_ok=True)

            logger.info(f"Initializing lists for {self.name} timing.")
    
            self.proc_timestamps = []
            self.get_wav_out = []
            self.get_spec = []
            self.spec_to_store = []
            self.to_device = []
            self.inference_time = []
            self.z_to_np = []
            self.z_to_store = []
            self.put_q_out = []
            self.proc_total_times = []

        if self.save_latent:
            self.latents_path = os.path.join(self.out_path, 'latents')
            os.makedirs(self.latents_path, exist_ok=True)
            logger.info(f"Latents directory: {self.latents_path}")
        if self.save_spec:
            self.specs_path = os.path.join(self.out_path, 'specs')
            os.makedirs(self.specs_path, exist_ok=True)
            logger.info(f"Spectrograms directory: {self.specs_path}")
        if self.save_stft:
            self.stfts_path = os.path.join(self.out_path, 'stft')
            logger.info(f"STFTs directory: {self.stfts_path}")
            os.makedirs(self.stfts_path, exist_ok=True)

        self.dropped_wav = []

        logger.info(f"{self.name} loading model: {self.model_path}")

        t = time.perf_counter_ns()
        self.model = VAE().eval().to(self.device)
        self.model.load_state(self.model_path)
        # self.model = torch.jit.load(self.model_path).eval().to(self.device)
        # self.model = torch.load(self.model_path).eval().to(self.device)
        torch.cuda.synchronize()
        load_model_time = (time.perf_counter_ns() - t) * 10**-6

        print(f"Time to load model: {load_model_time} ms")
        if self.time_opt:
            with open(os.path.join(self.timing_path, "load_model_time.txt"), "w") as text_file:
                text_file.write("%s" % load_model_time)
                text_file.close()

        t = time.perf_counter_ns()
        sample_input = torch.rand(size=(1,128,128), device=self.device)
        # self.model = torch.jit.optimize_for_inference(self.model)
        # Still have the high first run??? Why even though warmup in setup?
        with torch.no_grad():
            for _ in range(5):
                self.model(sample_input).to(self.device)
                torch.cuda.synchronize()
                
        warmup_time = (time.perf_counter_ns() - t) * 10**-6
        print(f"Time to warmup model: {warmup_time} ms")
        if self.time_opt:
            with open(os.path.join(self.timing_path, "warmup_time.txt"), "w") as text_file:
                text_file.write("%s" % warmup_time)
                text_file.close()

    def stop(self):
        """_summary_

        Returns:
            _type_: _description_
        """

        logger.info(f"{self.name} stopping.")

        if self.time_opt:
            keys = self.timing
            values = [self.proc_timestamps, self.get_wav_out, self.get_spec, self.spec_to_store, self.to_device, self.inference_time, self.z_to_np, self.z_to_store, self.put_q_out, self.proc_total_times]

            timing_dict = dict(zip(keys, values))
            df = pd.DataFrame.from_dict(timing_dict, orient='index').transpose()
            df.to_csv(os.path.join(self.timing_path, 'proc_timing_' + str(self.n_segs) + '.csv'), index=False, header=True)

        logger.info(f"{self.name} stopped.")

        return 0
        
    def runStep(self):
        """Run step for AVA AudioProcessor.
        """
        if self.done:
            pass

        self.proc_timestamps.append(time.perf_counter_ns())

        ids = self._checkInput()

        if ids is not None:
            t = time.perf_counter_ns()
            self.done = False
            try:
                t1 = time.perf_counter_ns()
                audio = self.client.getID(ids[0])
                t2 = time.perf_counter_ns()

                # Get only filename of segment w/o extension...do in acquire.py instead?
                fpath = Path(os.path.split(self.client.getID(ids[2]))[1]).stem
                fname =  '_'.join(fpath.split('_')[:-1])
                seg = fpath.split('_')[-1]
                
                spec, dt, f = get_spec(audio, self.params)
                if self.save_stft:
                    stft_path = os.path.join(self.stfts_path, fname)
                    os.makedirs(stft_path, exist_ok=True)
                    np.save(os.path.join(stft_path, str(seg) + '.npy'), spec)
                #
                # from ava.segment.amplitude_segmentation import get_onsets_offsets...
                #
                # # Calculate amplitude and smooth.
                # if p['softmax']:
                #     amps = softmax(spec, t=p['temperature'])
                # else:
                #     amps = np.sum(spec, axis=0)
                # amps = gaussian_filter(amps, p['smoothing_timescale']/dt)

                # power = sum(abs(audio))

                # amps = np.sum(spec, axis=0)
                # # amps = 10*np.log(silence_spec[1])

                # if amps.max() < self.params['amp_threshold']:
                #     seg = seg + str('_skip')
                t3 = time.time()
                spec = self._genSpec(0, self.win_len, audio, self.params['fs'], self.params)
                t4 = time.time()
                # if self.shoulder_spec:
                #     shoulder_spec = self._genSpec(self.shoulder, self.shoulder+len(audio), audio, , self.params)
                if self.save_spec:
                    spec_path = os.path.join(self.specs_path, fname)
                    os.makedirs(spec_path, exist_ok=True)
                    np.save(os.path.join(spec_path, str(seg) + '.npy'), spec)
                t5 = time.time()
                spec_obj_id = self.client.put(spec, 'spec' + str(self.seg_num))
                t6 = time.time()
                latents, t_dev, t_inf = self._runInference(spec)
                t7 = time.time()
                latents = latents[0].detach().cpu().numpy()
                t8 = time.time()
                if self.save_latent:
                    latent_path = os.path.join(self.latents_path, fname)
                    os.makedirs(latent_path, exist_ok=True)
                    np.save(os.path.join(latent_path, str(seg) + '.npy'), latents)
                t9 = time.time()
                latents_obj_id = self.client.put(latents, 'latents' + str(self.seg_num))
                t10 = time.time()
                # self.q_out.put([spec_obj_id, latents_obj_id, str(self.seg_num)])
                self.put_q_out.append((time.time() - t10)*1000.0)

                self.get_wav_out.append((t2 - t1)*1000.0)
                self.get_spec.append((t4 - t3)*1000.0)
                self.spec_to_store.append((t6 - t5)*1000.0)
                self.to_device.append(t_dev*1000.0)
                self.inference_time.append(t_inf*1000.0)
                self.z_to_np.append((t8 - t7)*1000.0)
                self.z_to_store.append((t10 - t9)*1000.0)

                self.seg_num += 1
                self.proc_total_times.append((time.time() - t)*1000.0)

            # Insert exceptions here...ERROR HANDLING, SEE ANNE'S ACTORS - from 1p demo
            except ObjectNotFoundError:
                logger.error(f"Processor: Audio {self.seg_num} unavailable from store, dropping")
                self.dropped_wav.append(self.seg_num)
                # self.q_out.put([1])
            except KeyError as e:
                logger.error(f"Processor: Key error... {e}")
                # Proceed at all costs
                self.dropped_wav.append(self.seg_num)
            except Exception as e:
                logger.error(f"Processor error: {type(e).__name__}: {e} during segment number {self.seg_num}")
                logger.info(traceback.format_exc())
                self.dropped_wav.append(self.seg_num)
            self.proc_total_times.append((time.perf_counter_ns() - t) * 10**-6)
        else:
            pass

        logger.info(f"Processor broke, avg time per segment: {np.mean(self.proc_total_times)}")
        logger.info(f"Processor got through {self.seg_num} segments.")

        if self.seg_num == self.n_segs:
            logger.error(f"Done processing all available data: {self.seg_num}")
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

    # def _skipSilence(self, audio, amp_threshold):
    #     '''
    #     '''

    def _genSpec(self, onset, offset, audio, fs, params, target_times=None):
        '''
        '''
        spec, _ = get_interp_spec(onset, offset, audio, params, fs, target_times=target_times, target_freqs=None, max_dur=None, remove_dc_offset=True, fill_value=None)
        
        #NOT WRITABLE ERROR:
        # audio_t = torch.Tensor(audio).to(self.device)

        # n_fft = params['nperseg']

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

        # transforms = TA.Spectrogram(
        #     n_fft=n_fft,
        #     win_length=params['win_len'],
        #     hop_length=params['noverlap'],
        #     center=True,
        #     power=2.0).to(self.device)

        # spec = transforms(audio_t).to(self.device)
            
        # transform = TV.Resize((params['height'], params['width'])).to(self.device)

        # spec_img = transform(spec.unsqueeze(0))

        return spec

    def _runInference(self, spec):
        '''
        '''
        t = time.perf_counter_ns()
        spec = torch.Tensor(spec).to(self.device)
        torch.cuda.synchronize()
        to_device = (time.perf_counter_ns() - t) * 10**-6
        with torch.no_grad():
            t = time.perf_counter_ns()
            output = self.model.encode(spec.unsqueeze(0))
            torch.cuda.synchronize()
            inf_time = (time.perf_counter_ns() - t) * 10**-6

        return output, to_device, inf_time

    # def _makeSpecMovie(self, fps, params, audio, fs):
    #     specs = []
    #     shoulder = params['shoulder']
    #     # Make spectrograms.
    #     dt = 1/fps
    #     onset = shoulder
    #     while onset + params['window_length'] < len(audio)/fs - shoulder:
    #         offset = onset + params['window_length']
    #         target_times = np.linspace(onset, offset, params['num_time_bins'])
    #         # Then make a spectrogram.
    #         spec, _ = self._genSpec(onset-shoulder, offset+shoulder, audio, params, \
    #                 fs=fs, target_times=target_times)
    #         specs.append(spec)
    #         onset += dt
    #     assert len(specs) > 0
    #     specs = np.stack(specs)

    #     if method in ['latent_nn', 're_umap']:
    #         # Make a DataLoader out of these spectrograms.
    #         loader = DataLoader(SimpleDataset(specs))
    #         # Get latent means.
    #         latent = self._runInference(loader)


    #     if method == 'latent_nn':
    #         # Get original latent and embeddings.
    #         original_embed = dc.request('latent_mean_umap')
    #         original_latent = dc.request('latent_means')
    #         # Find nearest neighbors in latent space to determine embeddings.
    #         new_embed = np.zeros((len(latent),2))
    #         for i in range(len(latent)):
    #             index = np.argmin([euclidean(latent[i], j) for j in original_latent])
    #             new_embed[i] = original_embed[index]

    # def _plot_specs(self, spec, latents):
    #     '''
    #     '''
    #     try:
    #         plt.imshow(spec)
    #         plt.imsave(os.path.join(self.spec_dir + str(self.fname) + str(self.seg_num) + '.png'))
    #         plt.close()
    #     except Empty as e:
    #         pass
    #     except Exception as e:
    #         # logger.error('Visual: Exception in get data: {}'.format(e))
    #         pass