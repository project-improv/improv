from PyQt5 import QtCore, QtGui, QtWidgets
from pyqtgraph import ImageView, PlotWidget
import numpy as np
from math import floor
import time
import pyqtgraph
from pyqtgraph import EllipseROI, PolyLineROI, ColorMap, ROI, LineSegmentROI
from queue import Empty
from matplotlib import cm
from matplotlib.colors import ListedColormap
from PyQt5 import QtGui,QtCore,QtWidgets
from PyQt5.QtGui import QColor, QPixmap
from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtWidgets import QMessageBox, QFileDialog
import guisetup
# import front_end
# from front_end import QRangeSlider
# from guisetup import frame_3


class extraSetup:
    def slider1(self): #testing implementability of sliders
        
        self.frame_3 = QtWidgets.QFrame(self.centralwidget)
        self.frame_3.setGeometry(QtCore.QRect(760, 20, 341, 241))
        self.frame_3.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_3.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_3.setObjectName("frame_3")
        self.slider = QtWidgets.QSlider(self.frame_3)
        self.slider.setGeometry(QtCore.QRect(30, 70, 131, 22))
        self.slider.setOrientation(QtCore.Qt.Horizontal)
        self.slider.setObjectName("slider")
        print("running slider")

    def slider2(self):
        self.slider2 = QRangeSlider(self.frame_3)
        self.slider2.setGeometry(QtCore.QRect(20, 100, 155, 50))
        # self.slider2.setGeometry(QtCore.QRect(55, 120, 155, 50))
        self.slider2.setObjectName("slider2") 


class QRangeSlider(QtWidgets.QWidget):
    
    rangeChanged = pyqtSignal(tuple, name='rangeChanged')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._minimum = 0
        self._maximum = 180

        self.min_max = 99
        self.max_max = 99

        self._layout = QtWidgets.QHBoxLayout()
        self._layout.setSpacing(0)
        self.setLayout(self._layout)

        self._min_slider = QtWidgets.QSlider(Qt.Horizontal)
        self._min_slider.setInvertedAppearance(True)

        self._max_slider = QtWidgets.QSlider(Qt.Horizontal)

        # install update handlers
        for slider in [self._min_slider, self._max_slider]:
            slider.blockSignals(True)
            slider.valueChanged.connect(self._value_changed)
            slider.rangeChanged.connect(self._update_layout)

            self._layout.addWidget(slider)

        # initialize to reasonable defaults
        self._min_slider.setValue(1 * self._min_slider.maximum())
        self._max_slider.setValue(1 * self._max_slider.maximum())

        self._update_layout()

    def _value_changed(self, *args):
        self._update_layout()
        self.rangeChanged.emit(self.range())

    def _update_layout(self, *args):
        for slider in [self._min_slider, self._max_slider]:
            slider.blockSignals(True)

        mid = floor((self._max_slider.value()-self._min_slider.value())/ 2)

        self.setMax_min(self._min_slider.maximum() + mid)
        self._min_slider.setValue(self._min_slider.value() + mid)
        self.setMax_max(self._max_slider.maximum() - mid)
        self._max_slider.setValue(self._max_slider.value() - mid)

        for slider in [self._min_slider, self._max_slider]:
            slider.blockSignals(False)

        self._layout.setStretch(0, self._min_slider.maximum())
        self._layout.setStretch(1, self._max_slider.maximum())

    def setMax_min(self, value):
        self._min_slider.setMaximum(value)
        self.min_max = value

    def setMax_max(self, value):
        self._max_slider.setMaximum(value)
        self.max_max = value

    def getMax_min(self):
        return self.min_max

    def getMax_max(self):
        return self.max_max

    def lowerSlider(self):
        return self._min_slider

    def upperSlider(self):
        return self._max_slider

    def range(self):
        return (self.getMax_min() - self._min_slider.value(), 180 - (self.getMax_max() - self._max_slider.value()))

    def setRange(self, lower, upper):
        for slider in [self._min_slider, self._max_slider]:
            slider.blockSignals(True)
        self._update_layout()