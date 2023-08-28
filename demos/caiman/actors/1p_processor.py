import time
import json
import os
import numpy as np
from improv.store import ObjectNotFoundError
from caiman.source_extraction.cnmf.online_cnmf import OnACID
from caiman.source_extraction.cnmf.params import CNMFParams
from caiman.utils.visualization import get_contours
from demos.sample_actors.process import CaimanProcessor
from queue import Empty
import traceback

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class OnePProcessor(CaimanProcessor):
    """Using 1p method from Caiman"""

    def __init__(self, *args, init_filename=None, config_file=None):
        super().__init__(*args, init_filename=init_filename, config_file=config_file)

    def setup(self):
        """Using #2 method from the realtime demo, with short init
        and online processing with OnACID-E
        """
        logger.info("Running setup for " + self.name)
        self.done = False
        self.dropped_frames = []
        self.coords = None
        self.ests = None
        self.A = None
        self.num = 0
        self.saving = False

        self.params = self.loadParams(param_file=self.param_file)

        # MUST include inital set of frames
        logger.info(self.params["fnames"])

        self.opts = CNMFParams(params_dict=self.params)
        self.onAc = OnACID(params=self.opts)
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

    def runStep(self):
        """Run process. Runs once per frame.
        Output is a location in the DS to continually
        place the Estimates results, with ref number that
        corresponds to the frame number
        """
        super().runStep()

    def putEstimates(self):
        """Put whatever estimates we currently have
        into the data store
        """
        t = time.time()
        A = self.onAc.estimates.Ab
        before = self.params["init_batch"]
        C = self.onAc.estimates.C_on[
            : self.onAc.N, before : self.frame_number + before
        ]  # .get_ordered()
        t2 = time.time()

        image = self.makeImage()
        t3 = time.time()
        dims = image.shape
        self._updateCoords(A, dims, C.shape[0])
        t4 = time.time()

        ids = []
        ids.append([self.client.put(self.coords, "coords" + str(self.frame_number)),
                "coords" + str(self.frame_number),])
        ids.append([self.client.put(image, "proc_image" + str(self.frame_number)),
                "proc_image" + str(self.frame_number),])
        ids.append([self.client.put(C, "C" + str(self.frame_number)),
                "C" + str(self.frame_number),])
        ids.append([self.frame_number, str(self.frame_number)])

        self.put(ids)

        t6 = time.time()

        self.putAnalysis_time.append([time.time() - t, t2 - t, t3 - t2, t4 - t3, t6 - t4])

    def _updateCoords(self, A, dims, num):
        """See if we need to recalculate the coords
        Also see if we need to add components
        """
        if self.coords is None:  # initial calculation
            self.A = A
            self.coords = get_contours(A, dims)
            self.num = num

        elif (self.num < num):  
            # np.shape(A)[1] > np.shape(self.A)[1] and self.frame_number % 200 == 0:
            # Only recalc if we have new components
            # FIXME: Since this is only for viz, only do this every 100 frames
            # TODO: maybe only recalc coords that are new?
            logger.info("Recomputing spatial contours")
            self.A = A
            self.coords = get_contours(A, dims)
            self.counter += 1

    def makeImage(self):
        """Create image data for visualiation
        Using caiman code here
        -- easier way to compute this?
        """
        mn = self.onAc.M - self.onAc.N
        image = self.frame.copy()
        try:
            components = (self.onAc.estimates.Ab[:, mn:]
                          .dot(self.onAc.estimates.C_on[mn : self.onAc.M, (self.frame_number - 1)])
                          .reshape(self.onAc.estimates.dims, order="F"))

            ssub_B = self.onAc.params.get("init", "ssub_B") * self.onAc.params.get("init", "ssub")
            if ssub_B == 1:
                B = (self.onAc.estimates.W.dot((image - components).flatten(order="F") 
                                               - self.onAc.estimates.b0) + self.onAc.estimates.b0)
                background = B.reshape(self.onAc.estimates.dims, order="F")
            else:
                bc2 = self.onAc.estimates.downscale_matrix.dot(
                    (image - components).flatten(order="F") - self.onAc.estimates.b0)
                background = (self.onAc.estimates.b0 + self.onAc.estimates.upscale_matrix.dot(
                        self.onAc.estimates.W.dot(bc2))).reshape(self.onAc.estimates.dims, order="F")

            image = ((components + background) - self.onAc.bnd_Y[0]) / np.diff(self.onAc.bnd_Y)
            image = np.minimum((image * 255.0), 255).astype("u1")
        except ValueError as ve:
            logger.info("ValueError: {0}".format(ve))

        # cor_frame = (self.frame - self.onAc.bnd_Y[0])/np.diff(self.onAc.bnd_Y)

        return image


class NaNFrameException(Exception):
    pass
