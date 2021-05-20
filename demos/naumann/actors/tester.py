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
import naumann_outline, lineplot, modelfitting

class Basic(QtGui.QMainWindow, naumann_outline.Ui_MainWindow, lineplot.lineplot, modelfitting.modelFitting):
    def __init__(self, type, parent=None):
        # ''' Setup GUI
        
        super(Basic, self).__init__(parent)
        self.setupUi(self)

        # type 1 = all components included
        if type == 1:
            self.makelineplot(self)

        # type 2 = only center chart
        if type == 2:
            self.makemodelfit(self)
        # type 3 = no polar plots
        if type == 3:
            self.makelineplot(self)
            self.makemodelfit(self)
            
    

        self.setup_standard(self)

        
from pyqtgraph import ImageView, PlotWidget
app = QApplication(sys.argv)
wind = Basic(type=1)
wind.show()
app.exec()