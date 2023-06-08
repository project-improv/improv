from improv.actor import Actor, Signal
from PyQt5 import QtWidgets
import numpy as np
from queue import Empty
from .front_end import FrontEnd
import logging
import traceback

logger = logging.getLogger(__name__)


class Visual(Actor):
    """Class used to run a GUI + Visual as a single Actor"""

    def setup(self, visual):
        # self.visual is CaimanVisual
        self.visual = visual
        self.visual.setup()
        logger.info("Running setup for " + self.name)

    def run(self):
        logger.info("Loading FrontEnd")
        self.app = QtWidgets.QApplication([])
        self.viewer = FrontEnd(self.visual, self.q_comm, self.q_sig)
        self.viewer.show()
        logger.info("GUI ready")
        self.q_comm.put([Signal.ready()])
        self.visual.q_comm.put([Signal.ready()])
        self.app.exec_()
        logger.info("Done running GUI")


class CaimanVisual(Actor):
    """Class for displaying data from caiman processor"""

    def __init__(self, *args, showConnectivity=False):
        super().__init__(*args)

    def setup(self):
        self.data = np.empty((0,0))
        self.bw_data = np.empty((0,0))

    def run(self):
        pass  # NOTE: Special case here, tied to GUI

    def getData(self):
        try:
            res = self.q_in.get(timeout=0.0005)
            #self.links['bq_out'].put()
            bw_res = self.links['bw_in'].get(timeout=0.0005)
            self.data = self.client.getID(res[1])
            self.bw_data = self.client.getID(bw_res[1])
        except Empty as e:
            pass
        except Exception as e:
            logger.error('Visual: Exception in get data: {}'.format(e))
            logger.error(traceback.format_exc())
