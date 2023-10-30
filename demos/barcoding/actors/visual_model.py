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
import cv2

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
        self.select_color = None
        self.coords = None
        self.w = None
        self.weight = None
        self.LL = None
        self.draw = True
        self.i2 = None
        self.total_times = []
        self.timestamp = []
        self.image = None
        self.neuron_coords = None
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
                    self.barcode_category,
                    self.color,
                    self.coords,
                    self.allStims,
                    self.image
                ) = self.client.getList(ids[:-1])
                self.total_times.append([time.time(), time.time() - t])
            self.timestamp.append([time.time(), self.frame_num])

            #logger.info("what is the barcode_category here? {0}".format(self.barcode_category[0]))
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
            self.barcode_out = [self.selectedBarcode, self.barcode]
        else:
            self.barcode_out = None

        if self.barcode_category is not None:
            #logger.info("ok let's see {0}".format(self.barcode_category))
            self.barcode_index_record = (self.barcode_category[0])['index_record']
            self.barcode_bytes_record = (self.barcode_category[0])['bytes_record']

        else:
            self.barcode_index_record = None
            self.barcode_bytes_record = None

  
        if self.frame_num > self.window:
            # self.Cx = self.Cx[-self.window:]
            self.C = self.C[:, -len(self.Cx) :]
            self.Cpop = self.Cpop[-len(self.Cx) :]

        return (
            self.Cx,
            self.C[self.selectedNeuron, :],
            self.Cpop,
            self.barcode_out,
            self.barcode_index_record,
            self.barcode_bytes_record,
        ) 

    def getFrames(self):
        """Return the raw and colored frames for display"""
        if self.raw is not None and self.raw.shape[0] > self.raw.shape[1]:
            self.raw = np.rot90(self.raw, 1)
        if self.color is not None and self.color.shape[0] > self.color.shape[1]:
            self.color = np.rot90(self.color, 1)
        if self.select_color is not None and self.select_color.shape[0] > self.select_color.shape[1]:
            self.select_color = np.rot90(self.select_color, 1)
        return self.raw, self.color, self.select_color 

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


    def select_color_neurons(self, user_select_barcode):
        neurons_locs = []
        neurons_index = []
        str_select_barcode = np.array2string(user_select_barcode)
        com_list = []
        com = np.array([o["CoM"] for o in self.coords])
        if str_select_barcode in self.barcode_index_record:
            neurons_index = self.barcode_index_record[str_select_barcode]
            self.select_color = self.plotColorsSelectBarcode(neurons_index, user_select_barcode)
            neuron_locs = [
                np.array([self.raw.shape[0] - com[i][0], self.raw.shape[1] - com[i][1]]) for i in neurons_index
            ]
        logger.info("test if i can get the index? {0}".format(neurons_index))
        logger.info("did i get the correct com list? {0}".format(neuron_locs))
        return neurons_locs, neurons_index


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
        if self.coords is not None:
            com = [o["CoM"] for o in self.coords]
            # first = [np.array([self.raw.shape[0]-com[0][1], com[0][0]])]
            first = [
                np.array([self.raw.shape[0] - com[0][0], self.raw.shape[1] - com[0][1]])
            ]
            # first = [com[0]]
        return first


    def plotColorsSelectBarcode(self, neurons_index, user_select_barcode):
        """Computes colored nicer background+components frame"""
        logger.info("start the color")
        t = time.time()
        image = self.image
        color = np.stack([image, image, image, image], axis=-1).astype(np.uint8).copy()
        color[..., 3] = 255
        # TODO: don't stack image each time?
        if self.coords is not None:
            # to delete: For checking if it can get the correct index
            neurons_index = neurons_index[1:3]
            #Problem is here.
            try:
                self.neuron_coords = [o["coordinates"] for o in self.coords]
            except Exception as e:
                logger.error("errors when indicing coords with selected neurons in barcode. {0}".format(e))
            count = 0
            try:
                for i, c in enumerate(self.neuron_coords):
                    if count >= np.shape(neurons_index)[0]:
                        break
                    elif i != neurons_index[count]:
                        continue
                    else:
                        assert(np.isin(i, neurons_index))
                        count += 1
                        # c = np.array(c)
                        ind = c[~np.isnan(c).any(axis=1)].astype(int)
                        logger.info("check7")
                        # TODO: Compute all colors simultaneously! then index in...
                        cv2.fillConvexPoly(
                            color, ind, self._defineColor(user_select_barcode)
                        )

            except Exception as e:
                logger.error("Exception happens during get the color, {0}".format(e))
                        

        # TODO: keep list of neural colors. Compute tuning colors and IF NEW, fill ConvexPoly.
        logger.info("ok let's see what's the color {0}".format(np.shape(color)))
        return color
    
    def _defineColor(self, user_select_barcode):
        """ind identifies the neuron by number"""
        ests = self.barcode
        # ests = self.tune_k[0]
        if ests is not None and user_select_barcode is not None:
            try:
                return self.manual_Color_Sum(user_select_barcode)
            except ValueError as e:
                logger.error("Value error here: {0}".format(e))
                return (255, 255, 255, 0)
            except Exception:
                print("user select barcode is ", user_select_barcode)
        else:
            return (255, 255, 255, 50)
        

    def manual_Color_Sum(self, x):
        """x should be length 12 array for coloring
        or, for k coloring, length 8
        Using specific coloring scheme from Naumann lab
        """
        if x.shape[0] == 8:
            mat_weight = np.array(
                [
                    [1, 0.25, 0],
                    [0.75, 1, 0],
                    [0, 1, 0],
                    [0, 0.75, 1],
                    [0, 0.25, 1],
                    [0.25, 0, 1.0],
                    [1, 0, 1],
                    [1, 0, 0.25],
                ]
            )
        elif x.shape[0] == 12:
            mat_weight = np.array(
                [
                    [1, 0.25, 0],
                    [0.75, 1, 0],
                    [0, 2, 0],
                    [0, 0.75, 1],
                    [0, 0.25, 1],
                    [0.25, 0, 1.0],
                    [1, 0, 1],
                    [1, 0, 0.25],
                    [1, 0, 0],
                    [0, 0, 1],
                    [0, 0, 1],
                    [1, 0, 0],
                ]
            )
        else:
            logger.info("Wrong shape for this coloring function")
            return (255, 255, 255, 10)

        color = x @ mat_weight

        blend = 0.8
        thresh = 0.2
        thresh_max = blend * np.max(color)
        color = np.clip(color, thresh, thresh_max)
        color -= thresh
        color /= thresh_max
        color = np.nan_to_num(color)
        if color.any() and np.linalg.norm(color - np.ones(3)) > 0.35:
            color *= 255
            return (color[0], color[1], color[2], 255)
        else:
            return (255, 255, 255, 10)