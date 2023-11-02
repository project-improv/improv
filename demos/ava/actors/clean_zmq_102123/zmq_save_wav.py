import time
import os
import numpy as np
from scipy.io.wavfile import write

from improv.store import CannotGetObjectError, ObjectNotFoundError
from queue import Empty

from pathlib import Path
from queue import Empty

import traceback, warnings

from improv.actor import Actor, RunManager

import logging; logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class SaveWav(Actor):
    """Actor for saving audio.
    """
    def __init__(self, *args, audio_dir=None, fs=None, time_opt=None, timing=None, out_path=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.done = False

        self.audio_dir = audio_dir
        self.fs = fs

        self.out_path = out_path

        self.seg_num = 0

    def __str__(self):
        return f"Name: {self.name}, Data: {self.data}"

    def setup(self):
        
        logger.info(f"Running setup for {self.name}.")

        os.makedirs(self.audio_dir, exist_ok=True)
        
    def stop(self):
        """_summary_

        Returns:
            _type_: _description_
        """

        logger.info(f"{self.name} stopping.")

        logger.info(f"{self.name} stopped.")

        return 0
    
    def runStep(self):

        if self.done:
            pass

        ids = self._checkInput()

        if ids is not None:
            self.done = False
            try:
                audio = self.client.getID(ids[0])

                fname = os.path.join(self.audio_dir, str(self.seg_num) + '.wav')

                write(fname, self.fs, audio)

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

        self.seg_num += 1

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