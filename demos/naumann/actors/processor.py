import time
import pickle
import json
import cv2
import numpy as np
import scipy.sparse
from improv.store import Store, CannotGetObjectError, ObjectNotFoundError
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

        self.loadParams(param_file=self.param_file)
        self.params = self.client.get("params_dict")

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
        super().runStep()

class NaNFrameException(Exception):
    pass
