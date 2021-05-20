
from PyQt5 import QtCore, QtGui, QtWidgets
from pyqtgraph import ImageView, PlotWidget

class modelFitting:
    def makemodelfit(self):
        self.frame_4 = QtWidgets.QFrame(self.centralwidget)
        self.frame_4.setGeometry(QtCore.QRect(1074, 308, 543, 549))
        self.frame_4.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_4.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_4.setObjectName("frame_4")
        self.rawplot_3 = ImageView(self.frame_4)
        self.rawplot_3.setGeometry(QtCore.QRect(10, 10, 531, 537))
        self.rawplot_3.setObjectName("rawplot_3")