from PyQt5 import QtCore, QtGui, QtWidgets
from pyqtgraph import ImageView, PlotWidget
import pyqtgraph
from pyqtgraph import EllipseROI, PolyLineROI, ColorMap, ROI, LineSegmentROI
import numpy as np

class polarplot:
    def __init__(self):
        self.num = 8
        theta = np.linspace(0, (315/360)*2*np.pi, self.num)
        theta = np.append(theta,0)
        self.theta = theta
        radius = np.zeros(self.num+1)
        self.thresh_r = radius + 1
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)



    def bottomplot(self):
        self.frame_9 = QtWidgets.QFrame(self.centralwidget)
        self.frame_9.setGeometry(QtCore.QRect(619, 155, 105, 101))
        self.frame_9.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_9.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_9.setObjectName("frame_9")
        self.grplot_4 = PlotWidget(self.frame_9)
        self.grplot_4.setGeometry(QtCore.QRect(6, 6, 93, 89))
        self.grplot_4.setObjectName("grplot_4")

        radarsetup(self.grplot_4)
        self.polar2 = self.grplot_4.plot()
        self.polar2.setData(x, y)

    def topplot(self):
        self.frame_7 = QtWidgets.QFrame(self.centralwidget)
        self.frame_7.setGeometry(QtCore.QRect(618, 30, 107, 103))
        self.frame_7.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_7.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_7.setObjectName("frame_7")
        self.grplot_3 = PlotWidget(self.frame_7)
        self.grplot_3.setGeometry(QtCore.QRect(6, 6, 95, 89))
        self.grplot_3.setObjectName("grplot_3")

        radarsetup(self.grplot_3)
        self.polar1 = self.grplot_3.plot()
        self.polar1.setData(x, y)


    def targetplot(self):
        self.frame_10 = QtWidgets.QFrame(self.frame_3)
        self.frame_10.setGeometry(QtCore.QRect(194, 64, 133, 125))
        self.frame_10.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_10.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_10.setObjectName("frame_10")
        self.grplot_5 = PlotWidget(self.frame_10)
        self.grplot_5.setGeometry(QtCore.QRect(6, 6, 121, 113))
        self.grplot_5.setObjectName("grplot_5")

        radarsetup(self.grplot_5)
        targetplotradar(self.grplot_5)
        self.polar3 = self.grplot_5.plot()

    
# def plotcustomization(self):
    

def radarsetup(polar):
    polar.setAspectLocked(True)

    # Add polar grid lines
    polar.addLine(x=0, pen=0.2)
    polar.addLine(y=0, pen=0.2)
    for r in range(0, 4, 1):
        circle = pyqtgraph.QtGui.QGraphicsEllipseItem(-r, -r, r*2, r*2)
        circle.setPen(pyqtgraph.mkPen(0.1))
        polar.addItem(circle)
    polar.hideAxis('bottom')
    polar.hideAxis('left')

    # self.polar1 = polars[0].plot()
    # self.polar2 = polars[1].plot()


def targetplotradar(polar):
    for r in range(2, 12, 2):
            circle = pyqtgraph.QtGui.QGraphicsEllipseItem(-r, -r, r*2, r*2)
            circle.setPen(pyqtgraph.mkPen(0.1))
            polar.addItem(circle)
    