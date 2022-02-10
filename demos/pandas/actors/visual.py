from improv.actor import Actor, Spike
from PyQt5 import QtWidgets
from queue import Empty
from .front_end import FrontEnd


import logging; logger = logging.getLogger(__name__)

class PandasVisual(Actor):
    ''' Class used to run a GUI + Visual as a single Actor
        Intended to be setup with a visual actor, BasicCaimanVisual (below).
    '''

    def setup(self, visual):
        logger.info('Running setup for '+self.name)
        self.visual = visual
        self.visual.setup()

    def run(self):
        logger.info('Loading FrontEnd')
        self.app = QtWidgets.QApplication([])
        self.viewer = FrontEnd(self.visual, self.q_comm, self.q_sig)
        self.viewer.show()
        logger.info('GUI ready')
        self.q_comm.put([Spike.ready()])
        self.visual.q_comm.put([Spike.ready()])
        self.app.exec_()
        logger.info('Done running GUI')

class PandasCaimanVisual(Actor):
    ''' Class for displaying data from caiman processor
    '''

    def __init__(self, *args, showConnectivity=False):
        super().__init__(*args)

    def setup(self):
        ''' Setup
        '''
        self.Cx = None
        self.C = None
        self.tune = None
        self.raw = None
        self.raw_frame_number = 0
        self.color = None
        self.coords = None

        self.draw = True

        self.total_times = []
        self.timestamp = []

        self.window = 500

    def run(self):
        pass  # NOTE: Special case here, tied to GUI

    def getData(self):
        try:
            obj_id = self.links['raw_frame_queue'].get(timeout=0.0001)
            self.raw_frame_number = obj_id[0]
            self.raw = self.client.getID(obj_id[1])
        except Empty as e:
            pass
        except Exception as e:
            logger.error('Visual: Exception in get data: {}'.format(e))