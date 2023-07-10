import time
import pickle
import json
import cv2
import numpy as np
import scipy.sparse
from improv.store import StoreInterface, CannotGetObjectError, ObjectNotFoundError
from caiman.source_extraction import cnmf
from caiman.source_extraction.cnmf.utilities import detrend_df_f
from caiman.source_extraction.cnmf.online_cnmf import OnACID
from caiman.source_extraction.cnmf.params import CNMFParams
from caiman.motion_correction import motion_correct_iteration_fast, tile_and_correct
from caiman.utils.visualization import get_contours
import caiman as cm
from os.path import expanduser
import os
from queue import Empty
from demos.sample_actors.process import CaimanProcessor
import traceback

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Naumann_processor(CaimanProcessor):
    """Wraps CaImAn/OnACID functionality to
    interface with our pipeline.
    Uses code from caiman/source_extraction/cnmf/online_cnmf.py
    """

    def __init__(self, *args, init_filename="data/tbif_ex.h5", config_file=None):
        super().__init__(*args, init_filename, config_file)

    def setup(self):
        """Create OnACID object and initialize it
        (runs initialize online)
        """
        logger.info("Running setup for " + self.name)
        self.done = False
        self.dropped_frames = []
        self.coords = None
        self.ests = None
        self.A = None

        self.params = self.loadParams(param_file=self.param_file)

        # MUST include inital set of frames
        # TODO: Institute check here as requirement to Nexus

        self.opts = CNMFParams(params_dict=self.params)
        self.onAc = OnACID(params=self.opts)
        # TODO: Need to rewrite init online as well to receive individual frames.
        self.onAc.initialize_online()
        self.max_shifts_online = self.onAc.params.get("online", "max_shifts_online")

        self.fitframe_time = []
        self.putAnalysis_time = []
        self.procFrame_time = []  # aka t_motion
        self.detect_time = []
        self.shape_time = []
        self.flag = False
        self.total_times = []
        self.timestamp = []
        self.counter = 0

    def stop(self):
        super().stop()

        before = self.params["init_batch"]
        nb = self.onAc.params.get("init", "nb")
        np.savetxt("output/raw_C.txt",
            np.array(self.onAc.estimates.C_on[nb : self.onAc.M, 
                      before : self.frame_number + before]),)
        
        # with open('output/S.pk', 'wb') as f:
        #     init = self.params['init_batch']
        #     S = np.stack([osi.s[init:] for osi in self.onAc.estimates.OASISinstances])
        #     print('--------Final S shape: ', S.shape)
        #     pickle.dump(S, f)
        with open("output/A.pk", "wb") as f:
            nb = self.onAc.params.get("init", "nb")
            A = self.onAc.estimates.Ab[:, nb:]
            print(type(A))
            pickle.dump(A, f)



    def runStep(self):
        """Run process. Runs once per frame.
        Output is a location in the DS to continually
        place the Estimates results, with ref number that
        corresponds to the frame number (TODO)
        """
        # TODO: Error handling for if these parameters don't work
        # should implement in Config (?) or getting too complicated for users..

        # proc_params = self.client.get('params_dict')
        init = self.params["init_batch"]
        frame = self._checkFrames()

        if frame is not None:
            t = time.time()
            self.done = False
            try:
                self.frame = self.client.getID(frame[0][str(self.frame_number)])
                self.frame = self._processFrame(self.frame, self.frame_number + init)
                t2 = time.time()
                self._fitFrame(self.frame_number + init, self.frame.reshape(-1, order="F"))
                self.fitframe_time.append([time.time() - t2])
                self.putEstimates()
                self.timestamp.append([time.time(), self.frame_number])
            except ObjectNotFoundError:
                logger.error("Processor: Frame {} unavailable from store, droppping"
                             .format(self.frame_number))
                self.dropped_frames.append(self.frame_number)
                self.q_out.put([1])
            except KeyError as e:
                logger.error("Processor: Key error... {0}".format(e))
                # Proceed at all costs
                self.dropped_frames.append(self.frame_number)
            except Exception as e:
                logger.error("Processor error: {}: {} during frame number {}"
                             .format(type(e).__name__, e, self.frame_number))
                print(traceback.format_exc())
                self.dropped_frames.append(self.frame_number)
            self.frame_number += 1
            self.total_times.append(time.time() - t)
        else:
            pass

    def _processFrame(self, frame, frame_number):
        """Do some basic processing on a single frame
        Raises NaNFrameException if a frame contains NaN
        Returns the normalized/etc modified frame
        """
        t = time.time()
        if frame is None:
            raise ObjectNotFoundError
        if np.isnan(np.sum(frame)):
            raise NaNFrameException
        frame = frame.astype(np.float32)  # or require float32 from image acquistion
        if self.onAc.params.get("online", "ds_factor") > 1:
            frame = cv2.resize(frame, self.onAc.img_norm.shape[::-1])
            # TODO check for params, onAc componenets before calling, or except
        if self.onAc.params.get("online", "normalize"):
            frame -= self.onAc.img_min
        if self.onAc.params.get("online", "motion_correct"):
            try:
                templ = (self.onAc.estimates.Ab.dot(
                        self.onAc.estimates.C_on[: self.onAc.M, (frame_number - 1)])
                        .reshape(# self.onAc.estimates.C_on[:self.onAc.M, (frame_number-1)%self.onAc.window]).reshape(
                        self.onAc.params.get("data", "dims"),order="F",)* self.onAc.img_norm)
            except Exception as e:
                logger.error("Unknown exception {0}".format(e))
                raise Exception

            if self.onAc.params.get("motion", "pw_rigid"):
                frame_cor, shift = tile_and_correct(
                    frame,
                    templ,
                    self.onAc.params.motion["strides"],
                    self.onAc.params.motion["overlaps"],
                    self.onAc.params.motion["max_shifts"],
                    newoverlaps=None,
                    newstrides=None,
                    upsample_factor_grid=4,
                    upsample_factor_fft=10,
                    show_movie=False,
                    max_deviation_rigid=self.onAc.params.motion["max_deviation_rigid"],
                    add_to_movie=0,
                    shifts_opencv=True,
                    gSig_filt=None,
                    use_cuda=False,
                    border_nan="copy",
                )[:2]
            else:
                frame_cor, shift = motion_correct_iteration_fast(
                    frame, templ, self.max_shifts_online, self.max_shifts_online)
            self.onAc.estimates.shifts.append(shift)
        else:
            frame_cor = frame
        if self.onAc.params.get("online", "normalize"):
            frame_cor = frame_cor / self.onAc.img_norm
        self.procFrame_time.append([time.time() - t])
        return frame_cor

class NaNFrameException(Exception):
    pass
