from improv.actor import Actor, Signal
from PyQt5 import QtWidgets
from queue import Empty
from .front_end import FrontEnd
import logging
import traceback

logger = logging.getLogger(__name__)


class Visual(Actor):
    """Class used to run a GUI + Visual as a single Actor"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if "name" in kwargs:
            self.name = kwargs["name"]
        
        self.state = GUIState(self)

    def run(self):
        self.register_with_nexus()
        self.setup_logging()
        self.register_with_broker()
        self.setup_links()
        self.q_comm = self.links["q_comm"]
        self.q_sig = self.links["q_sig"]

        logger.info("Loading FrontEnd")
        self.app = QtWidgets.QApplication([])
        self.viewer = FrontEnd(self.state) #, self.q_comm, self.q_sig)
        self.viewer.show()
        logger.info("GUI ready")
        self.q_comm.put([Signal.ready()])
        # self.visual.q_comm.put([Signal.ready()])
        self.app.exec_()
        logger.info("Done running GUI")

class GUIState:
    def __init__(self, gui):
        self.gui = gui
        self.data = None
        self.bw_mu = None
        self.bw_L = None
        self.bw_dead_nodes = None
        self.bw_n_obs = None
        self.frame_num = None
    
    def getData(self):
        """Load data from dim reduction and bubblewrap, returns false on timeout"""
        try:
            bw_res = self.gui.links['bw_in'].get(timeout=0.0005)
            res = self.gui.q_in.get(timeout=0.0005)
            self.data = self.gui.client.getID(res[1])
            self.bw_L = self.gui.client.getID(bw_res[1][1])
            self.bw_mu = self.gui.client.getID(bw_res[1][2])
            self.bw_n_obs = self.gui.client.getID(bw_res[1][3])
            self.bw_dead_nodes = self.gui.client.getID(bw_res[1][6])
        except Empty as e:
            return False
        except Exception as e:
            logger.error('Visual: Exception in get data: {}'.format(e))
            logger.error(traceback.format_exc())
        return True
    
    def send(self, msg_list):
        self.gui.q_comm.put(msg_list)


class BWVisual(Actor):
    """Class for preprocessing data from bubblewrap processor"""

    def __init__(self, *args, showConnectivity=False, **kwargs):
        super().__init__(*args, **kwargs)
        if "name" in kwargs:
            self.name = kwargs["name"]

    def setup(self):
        self.data = None
        self.bw_L = None

    # def run(self):
    #     pass  # NOTE: Special case here, tied to GUI

    def getData(self):
        """Load data from dim reduction and bubblewrap, returns false on timeout"""
        try:
            bw_res = self.links['bw_in'].get(timeout=0.0005)
            res = self.q_in.get(timeout=0.0005)
            self.data = self.client.getID(res[1])
            self.bw_L = self.client.getID(bw_res[1][1])
            self.bw_mu = self.client.getID(bw_res[1][2])
            self.bw_n_obs = self.client.getID(bw_res[1][3])
            self.bw_dead_nodes = self.client.getID(bw_res[1][6])
        except Empty as e:
            return False
        except Exception as e:
            logger.error('Visual: Exception in get data: {}'.format(e))
            logger.error(traceback.format_exc())
        return True
