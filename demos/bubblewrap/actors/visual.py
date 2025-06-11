from improv.actor import Actor, RunManager
from PyQt5 import QtWidgets
from queue import Empty
from .front_end import FrontEnd
from improv.messaging import ActorSignalMsg
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
        self.setup_logging()
        self.register_with_nexus()
        self.state.add_logger(self.improv_logger)
        self.state.add_signal_check()

        self.register_with_broker()
        self.setup_links()
        self.q_comm = self.links["q_comm"]
        self.q_sig = self.links["q_sig"]

        self.improv_logger.info("Loading FrontEnd")
        self.app = QtWidgets.QApplication([])
        self.viewer = FrontEnd(self.state) 
        self.viewer.show()
        self.app.exec_()
        self.improv_logger.info("Done running GUI")

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
            self.logger.error('Visual: Exception in get data: {}'.format(e))
            self.logger.error(traceback.format_exc())
        return True
    
    def send(self, signal):
        actor_signal = ActorSignalMsg(
            self.gui.name,
            signal,
            f"Sending signal {signal} to nexus",
        )
        self.gui.q_comm.put(actor_signal)
        return self.gui.q_comm.get()

    def add_logger(self, logger):
        self.logger = logger
    
    def add_signal_check(self):
        # make function to perform intermittent signal checking from Nexus
        gui = self.gui
        rm = RunManager(
            gui.name, gui.actions, gui.links, gui.nexus_sig_port, self.logger
        )
        self.signal_check = rm.loop_logic


class BWVisual(Actor):
    """Class for preprocessing data from bubblewrap processor"""

    def __init__(self, *args, showConnectivity=False, **kwargs):
        super().__init__(*args, **kwargs)
        if "name" in kwargs:
            self.name = kwargs["name"]

    def setup(self):
        self.data = None
        self.bw_L = None

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
