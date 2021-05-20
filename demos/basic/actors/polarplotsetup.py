from PyQt5 import QtCore, QtGui, QtWidgets
from pyqtgraph import ImageView, PlotWidget
import pyqtgraph
from pyqtgraph import EllipseROI, PolyLineROI, ColorMap, ROI, LineSegmentROI
import numpy as np

class polarplot:

    framepolarList = []
    framelistbare= []
    # desired plots specified in createplots command
    # 0= bottomplot, 1 = topplot, 2 = targetplot
    
    def bottomplot(self):
        self.frame_9 = QtWidgets.QFrame(self.centralwidget)
        self.framepolarList.append(self.frame_9)
        self.framelistbare.append("frame_9")
        self.frame_9.setGeometry(QtCore.QRect(619, 155, 105, 101))

        self.grplot_4 = PlotWidget(self.frame_9)
        self.framepolarList.append(self.grplot_4)
        self.framelistbare.append("grplot_4")
        self.grplot_4.setGeometry(QtCore.QRect(6, 6, 93, 89))

        # self.grplot_4.setObjectName("grplot_4")
        
        self.polar2 = self.grplot_4.plot()
        self.polar2.setData(self.x, self.y)

    def topplot(self):
        self.frame_7 = QtWidgets.QFrame(self.centralwidget)
        self.framepolarList.append(self.frame_7)
        self.framelistbare.append("frame_7")
        self.frame_7.setGeometry(QtCore.QRect(618, 30, 107, 103))
        
        self.grplot_3 = PlotWidget(self.frame_7)
        self.framepolarList.append(self.grplot_3)
        self.framelistbare.append("grplot_3")
        self.grplot_3.setGeometry(QtCore.QRect(6, 6, 95, 89))
        # self.grplot_3.setObjectName("grplot_3")

        self.polar1 = self.grplot_3.plot()
        self.polar1.setData(self.x, self.y)


    def targetplot(self):
        self.frame_10 = QtWidgets.QFrame(self.frame_3)
        self.framepolarList.append(self.frame_10)
        self.framelistbare.append("frame_10")
        self.frame_10.setGeometry(QtCore.QRect(194, 64, 133, 125))
    
        self.grplot_5 = PlotWidget(self.frame_10)
        self.framepolarList.append(self.grplot_5)
        self.framelistbare.append("grplot_5")
        self.grplot_5.setGeometry(QtCore.QRect(6, 6, 121, 113))

        targetplotradar(self.grplot_5)
        self.polar3 = self.grplot_5.plot()
    
    def createpolarplot(self, type):
        
        # framelistbare= ['frame_9','grplot_4', 'frame_7', "grplot_3", "frame_10", "grplot_5"]

        self.num = 8
        theta = np.linspace(0, (315/360)*2*np.pi, self.num)
        theta = np.append(theta,0)
        self.theta = theta
        radius = np.zeros(self.num+1)
        self.thresh_r = radius + 1

        self.x = np.zeros(self.num+1) * np.cos(self.theta)
        self.y = np.zeros(self.num+1) * np.sin(self.theta)

        if 0 in type:
            self.topplot()
        if 1 in type: 
            self.bottomplot()
        if 2 in type:
            self.targetplot()
        
        print("length: ", len(self.framelistbare))

        for x in range(len(type)):
            print(x)
            self.framepolarList[2*x].setFrameShape(QtWidgets.QFrame.StyledPanel)
            self.framepolarList[2*x].setFrameShadow(QtWidgets.QFrame.Raised)
            self.framepolarList[2*x].setObjectName(self.framelistbare[2*x])
            self.framepolarList[(2*x)+1].setObjectName(self.framelistbare[(2*x)+1])
            radarsetup(self.framepolarList[2*x+1])

        
        # x = radius * np.cos(theta)
        # y = radius * np.sin(theta)

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



def targetplotradar(polar):
    for r in range(2, 12, 2):
            circle = pyqtgraph.QtGui.QGraphicsEllipseItem(-r, -r, r*2, r*2)
            circle.setPen(pyqtgraph.mkPen(0.1))
            polar.addItem(circle)



    