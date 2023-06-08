import pyqtgraph
import numpy as np
import matplotlib.pylab as plt
import matplotlib

matplotlib.use("Qt5Agg")

from PyQt5 import QtGui, QtWidgets, QtCore
from PyQt5.QtGui import QColor
from . import improv_bubble
from improv.actor import Signal

from queue import Empty
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt5.QtWidgets import QGridLayout, QMessageBox

import logging
import traceback

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class FrontEnd(QtWidgets.QMainWindow, improv_bubble.Ui_MainWindow):
    def __init__(self, visual, comm, q_sig, parent=None):
        """Setup GUI
        Setup and start Nexus controls
        """
        logger.info("Setup and start Nexus controls")
        self.visual = visual
        self.comm = comm  # Link back to Nexus for transmitting signals
        self.q_sig = q_sig
        self.prev = 0
        self.n = 300

        pyqtgraph.setConfigOption("background", QColor(100, 100, 100))

        super(FrontEnd, self).__init__(parent)
        self.setupUi(self)
        pyqtgraph.setConfigOptions(leftButtonPan=True)

        self.scatter = pyqtgraph.ScatterPlotItem(
                size=10, brush=pyqtgraph.mkBrush(255, 255, 255, 120)
            )
        self.line = pyqtgraph.PlotDataItem(size=10)

        # Setup button
        self.pushButton.clicked.connect(_call(self._setup))

        # Run button
        self.pushButton_2.clicked.connect(_call(self._runProcess))
        self.pushButton_2.clicked.connect(_call(self.update)) # Tell Nexus to start
        self.dat = np.empty((0,2))
        self.prev_switch = None

    def update(self):
        self.visual.getData()
        if self.visual.data.size != 0:
            self.updateVisual()
        QtCore.QTimer.singleShot(10, self.update)

    def updateVisual(self):
        #ignoring (0,0)
        if not (self.visual.data[0][0] or self.visual.data[1][0]): return
        
        #append new data to self.dat
        newDat = np.array([self.visual.data[0][0], self.visual.data[1][0]])
        self.dat = np.vstack([self.dat, newDat])
        self.dat = self.dat[np.argsort(self.dat[:, 0])]


        if self.radioButton.isChecked():
            #check if button was recently pressed
            if self.prev_switch in ['l', None]:
                self.widget.removeItem(self.line)
                self.widget.addItem(self.scatter)
                self.prev_switch = 's'
            self.scatter.setData(pos=self.dat)
        elif self.radioButton_2.isChecked():
            if self.prev_switch in ['s', None]:
                self.widget.removeItem(self.scatter)
                self.widget.addItem(self.line)
                self.prev_switch = 'l'
            self.line.setData(self.dat)

    def _runProcess(self):
        logger.info("-------------------------   put run in comm")
        self.comm.put([Signal.run()])
        

    def _setup(self):
        logger.info("-------------------------   put setup in comm")
        self.comm.put([Signal.setup()])
        self.visual.setup()

    def closeEvent(self, event):
        """Clicked x/close on window
        Add confirmation for closing without saving
        """
        confirm = QMessageBox.question(
            self,
            "Message",
            "Quit without saving?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if confirm == QMessageBox.Yes:
            self.comm.put([Signal.quit()])
            # print('Visual broke, avg time per frame: ', np.mean(self.visual.total_times, axis=0))
            print("Visual got through ", self.visual.frame_num, " frames")
            # print('GUI avg time ', np.mean(self.total_times))
            event.accept()
        else:
            event.ignore()


def _call(fnc, *args, **kwargs):
    """Call handler for (external) events"""

    def _callback():
        return fnc(*args, **kwargs)

    return _callback
