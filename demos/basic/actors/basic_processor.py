import time
import pickle
import json
import cv2
import os
import numpy as np
import scipy.sparse
from improv.store import StoreInterface, CannotGetObjectError, ObjectNotFoundError
from caiman.source_extraction.cnmf.online_cnmf import OnACID
from caiman.source_extraction.cnmf.params import CNMFParams
from caiman.motion_correction import motion_correct_iteration_fast, tile_and_correct
from caiman.utils.visualization import get_contours
from os.path import expanduser
from queue import Empty

from improv.actor import RunManager
from demos.sample_actors.process import CaimanProcessor
import traceback

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class BasicProcessor(CaimanProcessor):
    """Uses CaimanProcessor from improv with additional custom
    code for the basic demo.
    """

    # TODO: Default data set for this. Ask for using Tolias from caiman...?
    def __init__(self, *args, init_filename="data/Tolias_mesoscope_2.hdf5", config_file=None):
        super().__init__(*args, init_filename=init_filename, config_file=config_file)

    def setup(self):
        super().setup()

    def stop(self):
        print("Processor broke, avg time per frame: ", np.mean(self.total_times, axis=0))
        print("Processor got through ", self.frame_number, " frames")
        if not os._exists("output"):
            try:
                os.makedirs("output")
            except:
                pass
        if not os._exists("output/timing"):
            try:
                os.makedirs("output/timing")
            except:
                pass
        np.savetxt("output/timing/process_frame_time.txt", np.array(self.total_times))
        np.savetxt("output/timing/process_timestamp.txt", np.array(self.timestamp))

        np.savetxt("output/timing/putAnalysis_time.txt", np.array(self.putAnalysis_time))
        np.savetxt("output/timing/procFrame_time.txt", np.array(self.procFrame_time))

        self.shape_time = np.array(self.onAc.t_shapes)
        self.detect_time = np.array(self.onAc.t_detect)

        np.savetxt("output/timing/fitframe_time.txt", np.array(self.fitframe_time))
        np.savetxt("output/timing/shape_time.txt", self.shape_time)
        np.savetxt("output/timing/detect_time.txt", self.detect_time)

    def runStep(self):
        """Run process. Runs once per frame.
        Output is a location in the DS to continually
        place the Estimates results, with ref number that
        corresponds to the frame number (TODO)
        """
        init = self.params["init_batch"]
        frame = self._checkFrames()

        if frame is not None:
            t = time.time()
            self.done = False
            try:
                self.frame = self.client.getID(frame[0][0])
                self.frame = self._processFrame(self.frame, self.frame_number + init)
                t2 = time.time()
                self._fitFrame(self.frame_number + init, self.frame.reshape(-1, order="F"))
                self.fitframe_time.append([time.time() - t2])
                self.putEstimates()
                self.timestamp.append([time.time(), self.frame_number])
            except ObjectNotFoundError:
                logger.error(
                    "Processor: Frame {} unavailable from store, droppping".format(self.frame_number))
                self.dropped_frames.append(self.frame_number)
                self.q_out.put([1])
            except KeyError as e:
                logger.error("Processor: Key error... {0}".format(e))
                # Proceed at all costs
                self.dropped_frames.append(self.frame_number)
            except Exception as e:
                logger.error("Processor error: {}: {} during frame number {}".format(
                    type(e).__name__, e, self.frame_number))
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
                templ = (self.onAc.estimates.Ab
                         .dot(self.onAc.estimates.C_on[: self.onAc.M, (frame_number - 1)])
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
    
    def putEstimates(self):
        """Put whatever estimates we currently have
        into the data store
        """
        t = time.time()
        nb = self.onAc.params.get("init", "nb")
        A = self.onAc.estimates.Ab[:, nb:]
        before = self.params["init_batch"] + (self.frame_number - 500 if self.frame_number > 500 else 0)
        C = self.onAc.estimates.C_on[nb : self.onAc.M, before : self.frame_number + before]  # .get_ordered()
        t2 = time.time()

        image = self.makeImage()
        # if self.frame_number == 1:
        #     np.savetxt('output/image.txt', np.array(image))
        t3 = time.time()
        dims = image.shape
        self._updateCoords(A, dims)
        t4 = time.time()

        ids = []
        ids.append([self.client.put(self.coords, "coords" + str(self.frame_number)),
                "coords" + str(self.frame_number),])
        ids.append([self.client.put(image, "proc_image" + str(self.frame_number)),
                "proc_image" + str(self.frame_number),])
        ids.append([self.client.put(C, "C" + str(self.frame_number)),
                "C" + str(self.frame_number),])
        ids.append([self.frame_number, str(self.frame_number)])

        t5 = time.time()

        # if self.frame_number %50 == 0:
        #     self.put(ids, save= [False, True, False, False])

        # else:
        # self.put(ids, save=[False]*4)

        self.put(ids)

        t6 = time.time()

        # self.q_comm.put([self.frame_number])

        self.putAnalysis_time.append([time.time() - t, t2 - t, t3 - t2, t4 - t3, t6 - t4])

    def _updateCoords(self, A, dims):
        """See if we need to recalculate the coords
        Also see if we need to add components
        """
        if self.coords is None:  # initial calculation
            self.A = A
            self.coords = get_contours(A, dims)

        elif np.shape(A)[1] > np.shape(self.A)[1] and self.frame_number % 200 == 0:
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
                .reshape(self.onAc.estimates.dims, order="F"))
            background = (self.onAc.estimates.Ab[:, :mn]
                .dot(self.onAc.estimates.C_on[:mn, (self.frame_number - 1)])
                .reshape(self.onAc.estimates.dims, order="F"))
            image = components + background # - self.onAc.bnd_Y[0]) / np.diff(self.onAc.bnd_Y)
            image = np.minimum((image * 255.0), 255).astype("u1")
        except ValueError as ve:
            logger.info("ValueError: {0}".format(ve))

        # cor_frame = (self.frame - self.onAc.bnd_Y[0])/np.diff(self.onAc.bnd_Y)

        return image


class NaNFrameException(Exception):
    pass
