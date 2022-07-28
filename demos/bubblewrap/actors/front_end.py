import pyqtgraph
import matplotlib.pylab as plt
import matplotlib
matplotlib.use('Qt5Agg')

from PyQt5 import QtGui, QtWidgets, QtCore
from PyQt5.QtGui import QColor
from . import improv_bubble
from improv.actor import Spike

from queue import Empty
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt5.QtWidgets import QGridLayout, QMessageBox

import logging; logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class MplCanvas(FigureCanvasQTAgg):

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super(MplCanvas, self).__init__(self.fig)

class FrontEnd(QtWidgets.QMainWindow, improv_bubble.Ui_Demo):
    def __init__(self, visual, comm, q_sig, parent=None):
        ''' Setup GUI
            Setup and start Nexus controls
        '''
        self.visual = visual
        self.comm = comm  # Link back to Nexus for transmitting signals
        self.q_sig = q_sig
        self.prev = 0

        pyqtgraph.setConfigOption('background', QColor(100, 100, 100))

        super(FrontEnd, self).__init__(parent)
        self.setupUi(self)
        pyqtgraph.setConfigOptions(leftButtonPan=True)
        self.sc = MplCanvas(self, width=5, height=4, dpi=100)

        #Setup button
        self.pushButton_4.clicked.connect(_call(self._setup))
        self.pushButton_4.clicked.connect(_call(self.started))
        
        #Run button
        self.pushButton_3.clicked.connect(_call(self._runProcess))
        self.pushButton_3.clicked.connect(_call(self.update)) # Tell Nexus to start

    def update(self):
        self.visual.getData()
        if(len(self.visual.data) != 0):
            if(self.prev == 0):
                self.loadVisual()
                self.prev = 1
            elif(self.prev != self.visual.data[0]):
                self.updateVisual()
                self.prev += 1

        QtCore.QTimer.singleShot(10, self.update)

    def updateVisual(self):
        if self.radioButton.isChecked():
            self.sc.axes.plot(self.visual.data[1][:, 0], self.visual.data[1][:, 1], color='gray', alpha=0.8)
        if self.radioButton_2.isChecked():
            self.sc.axes.scatter(self.visual.data[1][:, 0], self.visual.data[1][:, 1], color='gray', alpha=0.8)
        self.sc.fig.canvas.draw_idle()

    def loadVisual(self):
        ### 2D vdp oscillator
        if self.radioButton.isChecked():
            self.sc.axes.plot(self.visual.data[1][:, 0], self.visual.data[1][:, 1], color='gray', alpha=0.8)
        if self.radioButton_2.isChecked():
            self.sc.axes.scatter(self.visual.data[1][:, 0], self.visual.data[1][:, 1], color='gray', alpha=0.8)
        layout = QGridLayout()
        layout.addWidget(self.sc)
        self.frame.setLayout(layout)
        self.show()

    def started(self):
        try:
            signal = self.q_sig.get(timeout=0.000001)
            if(signal == Spike.started()):
                self.pushButton_3.setStyleSheet("background-color: rgb(255, 255, 255);")
                self.pushButton_4.setEnabled(False)
                self.pushButton_3.setEnabled(True)
            else:
                QtCore.QTimer.singleShot(10, self.started)
        except Empty as e:
            QtCore.QTimer.singleShot(10, self.started)
            pass
        except Exception as e:
            logger.error('Front End: Exception in get data: {}'.format(e))

    def _runProcess(self):
        self.comm.put([Spike.run()])
        logger.info('-------------------------   put run in comm')

    def _setup(self):
        self.comm.put([Spike.setup()])
        self.visual.setup()

    def closeEvent(self, event):
        '''Clicked x/close on window
            Add confirmation for closing without saving
        '''
        confirm = QMessageBox.question(self, 'Message', 'Quit without saving?',
                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if confirm == QMessageBox.Yes:
            self.comm.put([Spike.quit()])
            # print('Visual broke, avg time per frame: ', np.mean(self.visual.total_times, axis=0))
            print('Visual got through ', self.visual.frame_num, ' frames')
            # print('GUI avg time ', np.mean(self.total_times))
            event.accept()
        else: event.ignore()

def _call(fnc, *args, **kwargs):
    ''' Call handler for (external) events
    '''
    def _callback():
        return fnc(*args, **kwargs)
    return _callback