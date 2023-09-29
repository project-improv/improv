import time
import numpy as np
from improv.store import StoreInterface, ObjectNotFoundError
from scipy.spatial.distance import cdist
from math import floor
import colorsys
from PyQt5 import QtGui, QtWidgets
import pyqtgraph as pg
from .GUI import FrontEnd
import sys
from improv.actor import Actor, Signal
from queue import Empty
from collections import deque

import logging

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    handlers=[logging.FileHandler("example1.log"), logging.StreamHandler()],
)


class DisplayVisual(Actor):
    """Class used to run a GUI + Visual as a single Actor"""

    def run(self):
        logger.info("Loading FrontEnd")
        self.app = QtWidgets.QApplication([])
        self.rasp = FrontEnd(self.visual, self.q_comm)
        self.rasp.show()
        logger.info("GUI ready")
        self.q_comm.put([Signal.ready()])
        self.visual.q_comm.put([Signal.ready()])
        self.app.exec_()
        logger.info("Done running GUI")

    def setup(self, visual=None):
        logger.info("Running setup for " + self.name)
        self.visual = visual
        self.visual.setup()


class CaimanVisual(Actor):
    """Class for displaying data from caiman processor"""

    def __init__(self, *args, showConnectivity=True):
        super().__init__(*args)

        self.com1 = np.zeros(2)
        self.selectedNeuron = 0
        self.selectedBarcode = None
        self.frame_num = 0
        self.showConnectivity = showConnectivity

        self.stimStatus = dict()
        for i in range(8):
            self.stimStatus[i] = deque()

    def setup(self):
        self.Cx = None
        self.C = None
        self.barcode = None
        self.barcode_category = None
        self.raw = None
        self.color = None
        self.coords = None
        self.w = None
        self.weight = None
        self.LL = None
        self.draw = True
        self.sort_barcode_index = None
        self.i2 = None
        self.total_times = []
        self.timestamp = []

        self.window = 500

    def run(self):
        pass  # NOTE: Special case here, tied to GUI

    def getData(self):
        t = time.time()
        ids = None
        try:
            id = self.links["raw_frame_queue"].get(timeout=0.0001)
            self.raw_frame_number = list(id[0].keys())[0]
            self.raw = self.client.getID(id[0][self.raw_frame_number])
        except Empty as e:
            pass
        except Exception as e:
            logger.error("Visual: Exception in get raw data: {}".format(e))
        try:
            ids = self.q_in.get(timeout=0.0001)
            if ids is not None and ids[0] == 1:
                print("visual: missing frame")
                self.frame_num += 1
                self.total_times.append([time.time(), time.time() - t])
                raise Empty
            self.frame_num = ids[-1]
            if self.draw:
                (
                    self.Cx,
                    self.C,
                    self.Cpop,
                    self.barcode,
                    self.barcode_categoy,
                    self.color,
                    self.coords,
                    self.allStims,
                ) = self.client.getList(ids[:-1])
                logger.info(f"check the barcode category!!!!, {self.barcode_category}")
                self.total_times.append([time.time(), time.time() - t])
            self.timestamp.append([time.time(), self.frame_num])
            #logger.info("what is the coords here? {0}, {1}".format(np.shape(self.coords), self.coords))
        except Empty as e:
            pass
        except ObjectNotFoundError as e:
            logger.error("Object not found, continuing anyway...")
        except Exception as e: 
            logger.error("Visual: Exception in get frame data: {}".format(e))

    def getCurves(self):
        """Return the fluorescence traces and calculated tuning curves
        for the selected neuron as well as the population average
        Cx is the time (overall or window) as x axis
        C is indexed for selected neuron and Cpop is the population avg
        tune is a similar list to C
        """
        if self.barcode is not None:
            self.selectedBarcode = self.barcode[self.selectedNeuron]
            self.barcode_out = [self.selectedBarcode, self.barcode, self.barcode_category]
        else:
            self.barcode_out = None

        if self.frame_num > self.window:
            # self.Cx = self.Cx[-self.window:]
            self.C = self.C[:, -len(self.Cx) :]
            self.Cpop = self.Cpop[-len(self.Cx) :]
        logger.info('why there are nonobject, {0}'.format(self.barcode_out))

        return (
            self.Cx,
            self.C[self.selectedNeuron, :],
            self.Cpop,
            self.barcode_out,
        ) 

    def getFrames(self):
        """Return the raw and colored frames for display"""
        if self.raw is not None and self.raw.shape[0] > self.raw.shape[1]:
            self.raw = np.rot90(self.raw, 1)
        if self.color is not None and self.color.shape[0] > self.color.shape[1]:
            self.color = np.rot90(self.color, 1)

        return self.raw, self.color 

    def selectNeurons(self, x, y):
        """x and y are coordinates
        identifies which neuron is closest to this point
        and updates plotEstimates to use that neuron
        """
        neurons = [o["neuron_id"] - 1 for o in self.coords]
        com = np.array([o["CoM"] for o in self.coords])
        dist = cdist(com, [np.array([self.raw.shape[0] - x, self.raw.shape[1] - y])])
        if np.min(dist) < 50:
            selected = neurons[np.argmin(dist)]
            self.selectedNeuron = selected
            print("ID for selected neuron is :", selected)
            self.com1 = [
                np.array(
                    [
                        self.raw.shape[0] - com[selected][0],
                        self.raw.shape[1] - com[selected][1],
                    ]
                )
            ]
        else:
            logger.error("No neurons nearby where you clicked")
            self.com1 = [com[0]]
        return self.com1

    # def selectWeights(self, x, y):
        """x, y coordinates for neurons
        returns lines as 4 entry array: selected n_x, other_x, selected n_y, other_y
        """
        # translate back to C order of neurons from Caiman
        nid = self.i2[x]

        # highlight selected neuron
        com = np.array([o["CoM"] for o in self.coords])
        loc = [
            np.array([self.raw.shape[0] - com[nid][0], self.raw.shape[1] - com[nid][1]])
        ]

        # draw lines between it and all other in self.weight
        lines = np.zeros((18, 4))
        strengths = np.zeros(18)
        i = 0
        for n in np.nditer(self.i2):
            if n != nid and i < 9:
                if n < com.shape[0]:
                    ar = np.array(
                        [
                            self.raw.shape[0] - com[nid][0],
                            self.raw.shape[0] - com[n][0],
                            self.raw.shape[1] - com[nid][1],
                            self.raw.shape[1] - com[n][1],
                        ]
                    )
                    lines[i] = ar
                    #strengths[i] = self.w[nid][n]
                else:
                    strengths[i] = 0
                i += 1

        return loc, lines, strengths

    def selectNW(self, x, y):
        """x, y int
        lines 4 entry array: selected n_x, other_x, selected n_y, other_y
        """
        # translate back to C order of neurons
        nid = self.selectedNeuron
        logger.info("selected neuron {0}".format(nid))

        # highlight selected neuron
        com = np.array([o["CoM"] for o in self.coords])
        loc = [
            np.array([self.raw.shape[0] - com[nid][0], self.raw.shape[1] - com[nid][1]])
        ]

        # Rearrange barcode result
        if self.barcode is not None:
            self.selectedBarcode = self.barcode[self.selectedNeuron]

        lines = np.zeros((18, 4))
        strengths = np.zeros(18)
        i = 0

        return loc, lines, strengths
    
    def selectCategory(self, category_str):
        """x, y int
        lines 4 entry array: selected n_x, other_x, selected n_y, other_y
        """
        # translate back to C order of neurons
        index_record = self.barcode_category['index_record']
        byte_record = self.barcode_category['byte_record']
        selected_neuron_index = index_record[category_str]

        logger.info("selected barcode {0}".format(byte_record[category_str]))

        # highlight selected neuron
        com = np.array([o["CoM"] for o in self.coords])
        neuron_locs = [
            np.array([self.raw.shape[0] - com[i][0], self.raw.shape[1] - com[i][1]]) for i in selected_neuron_index
        ]
        
        return neuron_locs

    def getFirstSelect(self):
        first = None
        if self.coords:
            com = [o["CoM"] for o in self.coords]
            # first = [np.array([self.raw.shape[0]-com[0][1], com[0][0]])]
            first = [
                np.array([self.raw.shape[0] - com[0][0], self.raw.shape[1] - com[0][1]])
            ]
            # first = [com[0]]
        return first
