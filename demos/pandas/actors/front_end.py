from PyQt5 import QtGui, QtWidgets, QtCore
from PyQt5.QtGui import QColor
from . import improv_pandas
from improv.actor import Spike
import pyqtgraph
from queue import Empty
from demos.pandas.Pandas3D import live_QPanda3D
from QPanda3D.QPanda3DWidget import QPanda3DWidget
from PyQt5.QtWidgets import QGridLayout

import logging; logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class FrontEnd(QtWidgets.QMainWindow, improv_pandas.Ui_MainWindow):
    def __init__(self, visual, comm, q_sig, parent=None):
        ''' Setup GUI
            Setup and start Nexus controls
        '''
        self.visual = visual  # Visual class that provides plots and images
        self.comm = comm  # Link back to Nexus for transmitting signals
        self.q_sig = q_sig
        self.prev = 0
        self.total_times = []
        self.first = True
        self.world = None

        pyqtgraph.setConfigOption('background', QColor(100, 100, 100))
        super(FrontEnd, self).__init__(parent)
        self.setupUi(self)
        pyqtgraph.setConfigOptions(leftButtonPan=False)

        self.pushButton_2.clicked.connect(_call(self._setup))
        self.pushButton_2.clicked.connect(_call(self.started))
        self.pushButton_3.clicked.connect(_call(self._runProcess))
        self.pushButton_3.clicked.connect(_call(self.update)) # Tell Nexus to start

    def update(self):
        self.visual.getData()
        if(self.visual.raw != None):
            if(self.first):
                self.loadPandas()
                self.first = False
                self.prev = self.visual.raw_frame_number
            elif(self.prev != self.visual.raw_frame_number):
                self.prev = self.visual.raw_frame_number
                raw = self.visual.raw[2:len(self.visual.raw)]
                messenger.send("stimulus", raw)

        QtCore.QTimer.singleShot(10, self.update)

    def loadPandas(self):
        world = live_QPanda3D.PandaTest()
        world.get_size()
        raw = self.visual.raw[2:len(self.visual.raw)]
        world.createCard(raw[0], raw[1], raw[2], raw[3])
        pandaWidget = QPanda3DWidget(world)
        layout = QGridLayout()
        layout.addWidget(self.label, 0, 0)
        layout.addWidget(pandaWidget, 1, 0)
        self.frame.setLayout(layout)

    def started(self):
        try:
            signal = self.q_sig.get(timeout=0.000001)
            if(signal == Spike.started()):
                self.pushButton_3.setStyleSheet("background-color: rgb(255, 255, 255);")
                self.pushButton_2.setEnabled(False)
                self.pushButton_3.setEnabled(True)
            else:
                QtCore.QTimer.singleShot(10, self.started)
        except Empty as e:
            QtCore.QTimer.singleShot(10, self.started)
            pass
        except Exception as e:
            logger.error('Front End: Exception in get data: {}'.format(e))

    def _runProcess(self):
        # self.flag = True
        self.comm.put([Spike.run()])
        logger.info('-------------------------   put run in comm')

    def _setup(self):
        self.comm.put([Spike.setup()])
        self.visual.setup()

def _call(fnc, *args, **kwargs):
    ''' Call handler for (external) events
    '''
    def _callback():
        return fnc(*args, **kwargs)
    return _callback