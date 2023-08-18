import pyqtgraph
import numpy as np
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtGui import QColor
from . import improv_bubble
from improv.actor import Signal
from math import atan2, floor

from PyQt5.QtWidgets import QMessageBox

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

        pyqtgraph.setConfigOption("background", QColor(255, 255, 255))

        super(FrontEnd, self).__init__(parent)
        self.setupUi(self)
        pyqtgraph.setConfigOptions(leftButtonPan=True)

        self.plt = self.widget.getPlotItem()
        self.scatter = pyqtgraph.ScatterPlotItem(
                size = 10, brush=pyqtgraph.mkBrush(177, 177, 177),
                pen = pyqtgraph.mkPen(None),
                )
        self.data_red = np.empty((0,2))
        self.bw_center = pyqtgraph.ScatterPlotItem(
        size = 25, brush=pyqtgraph.mkBrush(0, 0, 0),
        pen = pyqtgraph.mkPen(None)
        )

        # Setup button
        self.pushButton.clicked.connect(_call(self._setup))

        # Run button
        self.pushButton_2.clicked.connect(_call(self._runProcess))
        self.pushButton_2.clicked.connect(_call(self.update)) # Tell Nexus to start

    def update(self):
        """Check if get data is successful, call plotting function and update GUI"""
        try:
            if self.visual.getData():
                self.plotBw()
        except Exception as e:
            logger.error('Front End Exception: {}'.format(e))
            logger.error(traceback.format_exc()) 
        QtCore.QTimer.singleShot(10, self.update)

    def plotBw(self):
        """Function for plotting dim reduced trajectories and bubbles"""
        self.plt.clear()
        # Dim reduced data plotting
        newDat = np.array([self.visual.data[0], self.visual.data[1]])
        self.data_red = np.vstack([self.data_red, newDat])
        self.scatter.setData(pos=self.data_red)
        self.plt.addItem(self.scatter)
        # bubble plotting
        for n in np.arange(self.visual.bw_L.shape[0]):
            if n not in self.visual.bw_dead_nodes:  #ignore dead nodes
                el = np.linalg.inv(self.visual.bw_L[n])
                sig = el.T @ el
                u,s,v = np.linalg.svd(sig)
                width, height = np.sqrt(s[0])*3, np.sqrt(s[1])*3
                angle = atan2(v[0,1],v[0,0])*360 / (2*np.pi)
                alpha_mat = 0.4
                x = self.visual.bw_mu[n,0]
                y = self.visual.bw_mu[n,1]
                el = QtWidgets.QGraphicsEllipseItem(x-(width/2), y-(height/2), width, height, self.plt)
                el.setBrush(pyqtgraph.mkBrush(QColor(237, 103, 19, int(alpha_mat/1*255))))
                el.setPen(pyqtgraph.mkPen(None))
                el.setTransformOriginPoint(x, y)
                el.setRotation(angle)
                self.plt.addItem(el)

        mask = np.ones(self.visual.bw_mu.shape[0], dtype=bool)
        mask[self.visual.bw_n_obs < .1] = False
        mask[self.visual.bw_dead_nodes] = False
        self.bw_center.setData(x = self.visual.bw_mu[mask, 0], y = self.visual.bw_mu[mask, 1])
        self.plt.addItem(self.bw_center)



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
