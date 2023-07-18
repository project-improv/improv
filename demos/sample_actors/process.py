import os
import time
import json
import cv2
import traceback

import numpy as np

from caiman.source_extraction import cnmf
from caiman.source_extraction.cnmf.online_cnmf import OnACID
from caiman.source_extraction.cnmf.params import CNMFParams
from caiman.motion_correction import motion_correct_iteration_fast, tile_and_correct
from caiman.utils.visualization import get_contours

from queue import Empty
from improv.actor import Actor
from improv.store import ObjectNotFoundError

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class CaimanProcessor(Actor):
    """Wraps CaImAn/OnACID functionality to
    interface with our pipeline.
    Uses code from caiman/source_extraction/cnmf/online_cnmf.py
    """

    def __init__(
        self, *args, init_filename="data/Tolias_mesoscope_2.hdf5", config_file=None
    ):
        super().__init__(*args)
        logger.info("initfile {}, config file {}".format(init_filename, config_file))
        self.param_file = config_file
        self.init_filename = init_filename
        self.frame_number = 0

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
        self.saving = True

        self.params = self.loadParams(param_file=self.param_file)
        # self.params = self.client.get('params_dict')

        # MUST include inital set of frames
        # TODO: Institute check here as requirement to Nexus
        logger.info(self.params["fnames"])

        self.opts = CNMFParams(params_dict=self.params)
        self.onAc = OnACID(params=self.opts)
        # TODO: Need to rewrite init online as well to receive individual frames.
        self.onAc.initialize_online(T=100000)
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
        print("Processor broke, avg time per frame: ", np.mean(self.total_times, axis=0))
        print("Processor got through ", self.frame_number, " frames")
        np.savetxt("output/timing/process_frame_time.txt", np.array(self.total_times))
        np.savetxt("output/timing/process_timestamp.txt", np.array(self.timestamp))

        self.shape_time = np.array(self.onAc.t_shapes)
        self.detect_time = np.array(self.onAc.t_detect)

        np.savetxt("output/timing/fitframe_time.txt", np.array(self.fitframe_time))
        np.savetxt("output/timing/shape_time.txt", self.shape_time)
        np.savetxt("output/timing/detect_time.txt", self.detect_time)

        np.savetxt("output/timing/putAnalysis_time.txt", np.array(self.putAnalysis_time))
        np.savetxt("output/timing/procFrame_time.txt", np.array(self.procFrame_time))

        print("Number of times coords updated ", self.counter)

        if self.onAc.estimates.OASISinstances is not None:
            try:
                init = self.params["init_batch"]
                S = np.stack([osi.s[init:] for osi in self.onAc.estimates.OASISinstances])
                np.savetxt("output/end_spikes.txt", S)
            except Exception as e:
                logger.error("Exception {}: {} during frame number {}"
                             .format(type(e).__name__, e, self.frame_number))
                print(traceback.format_exc())
        else:
            print("No OASIS")
        self.coords1 = [o["CoM"] for o in self.coords]
        print(self.coords1[0])
        print("type ", type(self.coords1[0]))
        np.savetxt("output/contours.txt", np.array(self.coords1))

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

    def loadParams(self, param_file=None):
        """Load parameters from file or 'defaults' into store
        TODO: accept user input from GUI
        This also effectively registers specific params
        that CaimanProcessor needs with Nexus
        TODO: Wrap init_filename into caiman params if params exist
        """
        cwd = os.getcwd() + "/"
        if param_file is not None:
            try:
                params_dict = self._load_params_from_file(param_file)
                params_dict["fnames"] = [cwd + self.init_filename]
            except Exception as e:
                logger.exception("File cannot be loaded. {0}".format(e))
        else:
            # defaults from demo scripts; CNMFParams does not set
            # each parameter needed by default (TODO change that?)
            # TODO add parameter validation inside Config
            params_dict = {
                "fnames": [cwd + self.init_filename],
                "fr": 2,
                "decay_time": 0.8,
                "gSig": (3, 3),
                "p": 1,
                "min_SNR": 1.5,
                "rval_thr": 1,
                "ds_factor": 1,
                "nb": 2,
                "motion_correct": True,
                "init_batch": 100,
                "init_method": "bare",
                "normalize": True,
                "sniper_mode": True,
                "K": 10,
                "epochs": 1,
                "max_shifts_online": 10,
                "pw_rigid": False,
                "dist_shape_update": True,
                "show_movie": False,
                "minibatch_shape": 100,
            }
        self.client.put(params_dict, "params_dict")

        return params_dict

    def _load_params_from_file(self, param_file):
        """Filehandler for loading caiman parameters
        TODO: Error handling, check params as loaded
        """
        params_dict = json.load(open(param_file, "r"))
        return params_dict

    def putEstimates(self):
        """Put whatever estimates we currently have
        into the data store
        """
        t = time.time()
        nb = self.onAc.params.get("init", "nb")
        A = self.onAc.estimates.Ab[:, nb:]
        before = self.params["init_batch"]  
        # self.frame_number-500 if self.frame_number > 500 else 0
        C = self.onAc.estimates.C_on[nb : self.onAc.M, before : self.frame_number + before]  # .get_ordered()
        t2 = time.time()
        t3 = time.time()

        image = self.makeImage()
        if self.frame_number == 1:
            np.savetxt("output/image.txt", np.array(image))
        t4 = time.time()
        dims = image.shape
        self._updateCoords(A, dims)
        t5 = time.time()

        ids = []
        ids.append(self.client.put(self.coords, "coords" + str(self.frame_number)))
        ids.append(self.client.put(image, "proc_image" + str(self.frame_number)))
        ids.append(self.client.put(C, "S" + str(self.frame_number)))
        ids.append(self.frame_number)
        t6 = time.time()

        self.q_out.put(ids)

        # self.q_comm.put([self.frame_number])

        self.putAnalysis_time.append([time.time() - t, t2 - t, t3 - t2, t4 - t3, t5 - t4, t6 - t5])

    def _checkFrames(self):
        """Check to see if we have frames for processing"""
        try:
            res = self.q_in.get(timeout=0.0005)
            return res
        # TODO: add'l error handling
        except Empty:
            # logger.info('No frames for processing')
            return None

    def _fitFrame(self, frame_number, frame):
        """Do the heavy lifting here. CNMF, etc
        Updates self.onAc.estimates
        """
        try:
            self.onAc.fit_next(frame_number, frame)
        except Exception as e:
            logger.error("Fit frame error! {}: {}".format(type(e).__name__, e))
            raise Exception

    def _updateCoords(self, A, dims):
        """See if we need to recalculate the coords
        Also see if we need to add components
        """
        if self.coords is None:  # initial calculation
            self.A = A
            self.coords = get_contours(A, dims)

        elif np.shape(A)[1] > np.shape(self.A)[1]:  # and self.frame_number % 50 == 0:
            # Only recalc if we have new components
            # FIXME: Since this is only for viz, only do this every 100 frames
            # TODO: maybe only recalc coords that are new?
            self.A = A
            self.coords = get_contours(A, dims)
            self.counter += 1

    def makeImage(self):
        """Create image data for visualiation
        Using caiman code here
        #TODO: move to MeanAnalysis class ?? Check timing if we move it!
            Other idea -- easier way to compute this?
        """
        mn = self.onAc.M - self.onAc.N
        image = None
        try:
            # components = self.onAc.estimates.Ab[:,mn:].dot(self.onAc.estimates.C_on[mn:self.onAc.M,(self.frame_number-1)%self.onAc.window]).reshape(self.onAc.dims, order='F')
            # background = self.onAc.estimates.Ab[:,:mn].dot(self.onAc.estimates.C_on[:mn,(self.frame_number-1)%self.onAc.window]).reshape(self.onAc.dims, order='F')
            components = (self.onAc.estimates.Ab[:, mn:]
                .dot(self.onAc.estimates.C_on[mn : self.onAc.M, (self.frame_number - 1)])
                .reshape(self.onAc.dims, order="F"))
            background = (self.onAc.estimates.Ab[:, :mn]
                .dot(self.onAc.estimates.C_on[:mn, (self.frame_number - 1)])
                .reshape(self.onAc.dims, order="F"))
            image = ((components + background) - self.onAc.bnd_Y[0]) / np.diff(self.onAc.bnd_Y)
            image = np.minimum((image * 255.0), 255).astype("u1")
        except ValueError as ve:
            logger.info("ValueError: {0}".format(ve))

        # cor_frame = (self.frame - self.onAc.bnd_Y[0])/np.diff(self.onAc.bnd_Y)

        return image


class NaNFrameException(Exception):
    pass
