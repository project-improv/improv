import time
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.io import wavfile
from scipy.io.wavfile import WavFileWarning

# For FolderAcquirer
from pathlib import Path
from queue import Empty

import traceback
import warnings

from improv.actor import Actor, RunManager

import logging; logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class AudioAcquirer(Actor):
    '''
    '''

    def __init__(self, *args, n_segs=None, folder=None, time_opt=True, timing=None, out_path=None, **kwargs):
    # def __init__(self, *args, folder=None, n_data=None, n_segs=None, window_len=None, prof_time=True, timing=None, out_path=None, **kwargs):
        super().__init__(*args, **kwargs)
        # TODO: Add params set in .yaml or input = file to load params after tuning offline
        # TODO: Add tuning online
        # with open(params, 'r') as p:
        #     self.params = params
        #     params.close()
        
        self.done = False

        self.seg_num = 0

        if folder is None:
            logger.error('Must specify folder of data.')
        else:
            self.path = Path(folder)
            if not self.path.exists() or not self.path.is_dir():
                raise AttributeError('Data folder {} does not exist.'.format(self.path))

        self.time_opt = time_opt
        if self.time_opt is True:
            self.timing = timing
            self.out_path = out_path

        self.n_segs = n_segs

    def setup(self):
        if self.time_opt is True:
            os.makedirs(self.out_path, exist_ok=True)

        # .wav files in specified dir
        self.files = [f.as_posix() for f in self.path.iterdir() if f.suffix == '.wav']
        # self.files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

        # def _get_wavs_from_dir(dir):
        #     """Return a sorted list of wave files from a directory."""
        #     return [os.path.join(dir, f) for f in sorted(os.listdir(dir)) if \
        #             _is_wav_file(f)]
        
        # if _is_wav_file(file):
        #     return audio
        # def _is_wav_file(filename):
        #     """Is the given filename a wave file?"""
        #     return len(filename) > 4 and filename[-4:] == '.wav'

        # In Visual Actor as well:
        # self.audio_dir = os.path.join(self.out_path, 'audio')
        # os.makedirs(self.audio_dir, exist_ok=True)

    def run(self):
        ''' Triggered at Run
        '''
        self.acq_total_times = []
        self.acq_timestamps = []
        self.get_wav_time = []
        self.put_wav_time = []
        self.put_out_time = []

        with RunManager(self.name, self.runAcquirer, self.setup, self.q_sig, self.q_comm, self.stop) as rm:
            print(rm)

        print('Acquire broke, avg time per segment:', np.mean(self.acq_total_times))
        print('Acquire got through', self.seg_num, ' segments')

    def stop(self):
        '''
        '''
        if self.time_opt is True:
            keys = self.timing
            values = [self.acq_total_times, self.acq_timestamps, self.get_wav_time, self.put_wav_time, self.put_out_time]
            timing_dict = dict(zip(keys, values))
            df = pd.DataFrame.from_dict(timing_dict, orient='index').transpose()
            df.to_csv(os.path.join(self.out_path, 'acq_timing_' + str(self.seg_num) + '.csv'), index=False, header=True)

        return 0

    def runAcquirer(self):
        '''
        '''
        if self.time_opt is True:
            self.acq_timestamps.append((time.time(), int(self.seg_num)))

        if self.done:
            pass

        elif self.seg_num != len(self.files):
            t = time.time()
            try:
                t1 = time.time()
                fs, audio = self.get_audio(self.files[self.seg_num])
                t2 = time.time()
                # self.plot_audio(audio, self.files[self.seg_num])
                audio_obj_id = self.client.put(audio, 'seg_num_' + str(self.seg_num))
                fs_obj_id = self.client.put(fs, 'seg_num_' + str(self.seg_num))
                fname_obj_id = self.client.put(self.files[self.seg_num], 'seg_num_' + str(self.seg_num))
                t3 = time.time()
                self.q_out.put([audio_obj_id, fs_obj_id, fname_obj_id, str(self.seg_num)])
                self.put_out_time.append((time.time() - t3)*1000.0)

                self.seg_num += 1

                self.acq_total_times.append((time.time() - t)*1000.0)
                self.get_wav_time.append((t2 - t1)*1000.0)
                self.put_wav_time.append((t3 - t2)*1000.0)

            except Exception as e:
                logger.error('Acquirer general exception: {}'.format(e))
            except IndexError as e:
                pass

        if self.seg_num == self.n_segs:
            logger.error('Done acquiring all available data: {}'.format(self.seg_num))
            self.data = None
            self.q_comm.put(None)
            self.done = True  # stay awake in case we get a shutdown signal

    def get_audio(self, file):
        '''
        '''
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=WavFileWarning)
            fs, audio = wavfile.read(file)
        return fs, audio
    

    def plot_audio(self, audio, fname):
        try:
            plt.plot(audio)
            plt.imsave(os.path.join(self.audio_dir + fname + str(self.seg_num) + '.png'))
            plt.close()
        except Empty as e:
            pass
        except Exception as e:
            # logger.error('Visual: Exception in get data: {}'.format(e))
            pass