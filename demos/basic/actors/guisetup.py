# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'improv_shell.ui'
#
# Created by: PyQt5 UI code generator 5.12.3
#
# WARNING! All changes made in this file will be lost!
from PyQt5 import QtCore, QtGui, QtWidgets
from pyqtgraph import ImageView, PlotWidget
import numpy as np

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1118, 652)
        MainWindow.setStyleSheet("QMainWindow { background-color: \'blue\'; }")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setStyleSheet("#centralwidget { background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:0, y2:1, stop:0 rgb(50, 50, 50), stop:1 rgba(80, 80, 80, 255)); }\n"
"#checkBox {color: black}\n"
"#label_2 {color: rgb(225, 230, 240)}\n"
"#label_3 {color: rgb(225, 230, 240)}\n"
"#label_4 {color: rgb(225, 230, 240)}\n"
"#label_5 {color: rgb(225, 230, 240)}\n"
"#frame {background-color: rgb(150, 160, 175);\n"
"border-style: outset;\n"
"border-width: 2px;\n"
"border-radius: 10px;\n"
"border-color: black;\n"
"padding: 6px}\n"
"#frame_3 {background-color: rgb(170, 185, 200);\n"
"border-style: outset;\n"
"border-width: 2px;\n"
"border-radius: 10px;\n"
"border-color: black;\n"
"padding: 6px}\n"
"#frame_2 {background: rgb(229, 229, 229);\n"
"border-style: outset;\n"
"border-width: 2px;\n"
"border-radius: 10px;\n"
"border-color: black;\n"
"padding: 1px}\n"
"#frame_4 {background: rgb(170, 185, 200);\n"
"border-style: outset;\n"
"border-width: 2px;\n"
"border-radius: 10px;\n"
"border-color: black;\n"
"padding: 1px}\n"
"#frame_5 {background: rgb(229, 229, 229);\n"
"border-style: outset;\n"
"border-width: 2px;\n"
"border-radius: 10px;\n"
"border-color: black;\n"
"padding: 6px}\n"
"#frame_6 {background: rgb(229, 229, 229);\n"
"border-style: outset;\n"
"border-width: 2px;\n"
"border-radius: 10px;\n"
"border-color: black;\n"
"padding: 6px}\n"
"#frame_8 {background: rgb(229, 229, 229);\n"
"border-style: outset;\n"
"border-width: 2px;\n"
"border-radius: 10px;\n"
"border-color: black;\n"
"padding: 6px}\n"
"#frame_7 {background: rgb(229, 229, 229);\n"
"border-style: outset;\n"
"border-width: 2px;\n"
"border-radius: 10px;\n"
"border-color: black;\n"
"padding: 6px}\n"
"#frame_9 {background: rgb(229, 229, 229);\n"
"border-style: outset;\n"
"border-width: 2px;\n"
"border-radius: 10px;\n"
"border-color: black;\n"
"padding: 0px}\n"
"#frame_10 {background: rgb(229, 229, 229);\n"
"border-style: outset;\n"
"border-width: 2px;\n"
"border-radius: 10px;\n"
"border-color: black;\n"
"padding: 6px}\n"
"#pushButton {\n"
"background-color: rgb(225, 230, 240);\n"
"color: black;\n"
"font: bold;\n"
"border-style: outset;\n"
"border-width: 2px;\n"
"border-radius: 10px;\n"
"border-color: black;\n"
"padding: 6px\n"
"}\n"
"#pushButton_2 {\n"
"background-color: rgb(225, 230, 240);\n"
"color: black;\n"
"font: bold;\n"
"border-style: outset;\n"
"border-width: 2px;\n"
"border-radius: 10px;\n"
"border-color: black;\n"
"padding: 6px\n"
"}\n"
"#pushButton_3 {\n"
"background-color: rgb(225, 230, 240);\n"
"color: black;\n"
"font: bold;\n"
"border-style: outset;\n"
"border-width: 2px;\n"
"border-radius: 10px;\n"
"border-color: black;\n"
"padding: 6px\n"
"}\n"
"#pushButton_4 {\n"
"background-color: rgb(200, 140, 140);\n"
"color: black;\n"
"font: bold;\n"
"border-style: outset;\n"
"border-width: 2px;\n"
"border-radius: 10px;\n"
"border-color: black;\n"
"padding: 6px\n"
"}")
    def setup_standard(self, MainWindow): 
        self.centralwidget.setObjectName("centralwidget")
        self.frame = QtWidgets.QFrame(self.centralwidget)
        self.frame.setGeometry(QtCore.QRect(10, 10, 151, 241))
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.pushButton = QtWidgets.QPushButton(self.frame)
        self.pushButton.setGeometry(QtCore.QRect(20, 10, 111, 51))
        self.pushButton.setObjectName("pushButton")
        self.pushButton_2 = QtWidgets.QPushButton(self.frame)
        self.pushButton_2.setGeometry(QtCore.QRect(20, 70, 111, 51))
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_3 = QtWidgets.QPushButton(self.frame)
        self.pushButton_3.setGeometry(QtCore.QRect(20, 130, 111, 51))
        self.pushButton_3.setObjectName("pushButton_3")
        self.checkBox = QtWidgets.QCheckBox(self.frame)
        self.checkBox.setGeometry(QtCore.QRect(20, 190, 111, 41))
        font = QtGui.QFont()
        font.setFamily("Helvetica Neue")
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.checkBox.setFont(font)
        self.checkBox.setObjectName("checkBox")
        # self.frame_3 = QtWidgets.QFrame(self.centralwidget)
        # self.frame_3.setGeometry(QtCore.QRect(760, 20, 341, 241))
        # self.frame_3.setFrameShape(QtWidgets.QFrame.StyledPanel)
        # self.frame_3.setFrameShadow(QtWidgets.QFrame.Raised)
        # self.frame_3.setObjectName("frame_3")
        # self.slider = QtWidgets.QSlider(self.frame_3)
        # self.slider.setGeometry(QtCore.QRect(30, 70, 131, 22))
        # self.slider.setOrientation(QtCore.Qt.Horizontal)
        # self.slider.setObjectName("slider")
        self.label = QtWidgets.QLabel(self.frame_3)
        self.label.setGeometry(QtCore.QRect(20, 20, 171, 21))
        font = QtGui.QFont()
        font.setFamily("Helvetica Neue")
        font.setPointSize(18)
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.pushButton_4 = QtWidgets.QPushButton(self.frame_3)
        self.pushButton_4.setGeometry(QtCore.QRect(30, 190, 115, 32))
        font = QtGui.QFont()
        font.setFamily("Helvetica Neue")
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.pushButton_4.setFont(font)
        self.pushButton_4.setObjectName("pushButton_4")
        self.label_6 = QtWidgets.QLabel(self.frame_3)
        self.label_6.setGeometry(QtCore.QRect(70, 90, 71, 21))
        font = QtGui.QFont()
        font.setFamily("Helvetica Neue")
        font.setPointSize(13)
        font.setBold(False)
        font.setWeight(50)
        self.label_6.setFont(font)
        self.label_6.setObjectName("label_6")
        self.label_7 = QtWidgets.QLabel(self.frame_3)
        self.label_7.setGeometry(QtCore.QRect(70, 140, 71, 21))
        font = QtGui.QFont()
        font.setFamily("Helvetica Neue")
        font.setPointSize(13)
        font.setBold(False)
        font.setWeight(50)
        self.label_7.setFont(font)
        self.label_7.setObjectName("label_7")
        # self.frame_10 = QtWidgets.QFrame(self.frame_3)
        # self.frame_10.setGeometry(QtCore.QRect(194, 64, 133, 125))
        # self.frame_10.setFrameShape(QtWidgets.QFrame.StyledPanel)
        # self.frame_10.setFrameShadow(QtWidgets.QFrame.Raised)
        # self.frame_10.setObjectName("frame_10")
        # self.grplot_5 = PlotWidget(self.frame_10)
        # self.grplot_5.setGeometry(QtCore.QRect(6, 6, 121, 113))
        # self.grplot_5.setObjectName("grplot_5")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(170, 10, 131, 21))
        font = QtGui.QFont()
        font.setFamily("Helvetica Neue")
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(170, 134, 131, 21))
        font = QtGui.QFont()
        font.setFamily("Helvetica Neue")
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.frame_6 = QtWidgets.QFrame(self.centralwidget)
        self.frame_6.setGeometry(QtCore.QRect(170, 30, 431, 101))
        self.frame_6.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_6.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_6.setObjectName("frame_6")
        self.grplot = PlotWidget(self.frame_6)
        self.grplot.setGeometry(QtCore.QRect(6, 6, 419, 89))
        self.grplot.setObjectName("grplot")
        self.frame_8 = QtWidgets.QFrame(self.centralwidget)
        self.frame_8.setGeometry(QtCore.QRect(170, 154, 431, 101))
        self.frame_8.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_8.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_8.setObjectName("frame_8")
        self.grplot_2 = PlotWidget(self.frame_8)
        self.grplot_2.setGeometry(QtCore.QRect(6, 6, 419, 89))
        self.grplot_2.setObjectName("grplot_2")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1118, 20))
        self.menubar.setObjectName("menubar")
        self.menuRASP_Display = QtWidgets.QMenu(self.menubar)
        self.menuRASP_Display.setObjectName("menuRASP_Display")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.menubar.addAction(self.menuRASP_Display.menuAction())
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
    
    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.pushButton.setText(_translate("MainWindow", "Load\n"
"Parameters"))
        self.pushButton_2.setText(_translate("MainWindow", "Setup"))
        self.pushButton_3.setText(_translate("MainWindow", "Run"))
        self.checkBox.setText(_translate("MainWindow", " Live Update"))
        self.label.setText(_translate("MainWindow", "Targeting Selection"))
        self.pushButton_4.setText(_translate("MainWindow", "Stimulate"))
        self.label_6.setText(_translate("MainWindow", "Threshold"))
        self.label_7.setText(_translate("MainWindow", "Direction"))
        self.label_2.setText(_translate("MainWindow", "Population Average"))
        self.label_3.setText(_translate("MainWindow", "Selected Neuron"))
        self.menuRASP_Display.setTitle(_translate("MainWindow", "Nexus Display"))

    def preppolarplots(self):

        # self.checkBox.setChecked(True)
        # self.draw = True

        # #init line plot
        # self.flag = True

        # self.c1 = self.grplot.plot(clipToView=True)
        # self.c2 = self.grplot_2.plot()
        # grplot = [self.grplot, self.grplot_2]
        # for plt in grplot:
        #     plt.getAxis('bottom').setTickSpacing(major=50, minor=50)
        # #    plt.setLabel('bottom', "Frames")
        # #    plt.setLabel('left', "Temporal traces")
        # self.updateLines()
        # self.activePlot = 'r'

        self.num = 8
        theta = np.linspace(0, (315/360)*2*np.pi, self.num)
        theta = np.append(theta,0)
        self.theta = theta
        radius = np.zeros(self.num+1)
        self.thresh_r = radius + 1
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)

from pyqtgraph import PlotWidget

    