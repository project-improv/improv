import sys
import pyqtgraph
from pyqtgraph import EllipseROI, PolyLineROI, ColorMap, ROI, LineSegmentROI
from queue import Empty
from matplotlib import cm
from matplotlib.colors import ListedColormap    
from PyQt5 import QtGui,QtCore,QtWidgets
from PyQt5.QtGui import QColor, QPixmap
from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtWidgets import QMessageBox, QFileDialog
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QLabel
from PyQt5.QtWidgets import QWidget
# import buildwindow
import rawplotsetup
import polarplotsetup
import guisetup, guistandard, extrasetup

class Basic(QtGui.QMainWindow, guisetup.Ui_MainWindow, rawplotsetup.rawplot, polarplotsetup.polarplot, extrasetup.extraSetup):
    def __init__(self, parent=None):
        super(Basic, self).__init__(parent)
        self.setupUi(self)
        self.slider1()
        self.slider2()
        self.makebottomright(self)
        self.makebottomcenter(self)
        self.makebottomleft()
        self.bottomplot()
        self.topplot()
        self.targetplot()
        # self.radarsetup()
        self.setup_standard(self)
        

from pyqtgraph import ImageView, PlotWidget
app = QApplication(sys.argv)
wind = Basic(None)
wind.show()
app.exec()