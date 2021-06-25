# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'improv_viz_stim.ui'
#
# Created by: PyQt5 UI code generator 5.12.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1633, 909)
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
        self.centralwidget.setObjectName("centralwidget")
        self.frame_2 = QtWidgets.QFrame(self.centralwidget)
        self.frame_2.setGeometry(QtCore.QRect(16, 308, 441, 521))
        self.frame_2.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_2.setObjectName("frame_2")
        self.rawplot = ImageView(self.frame_2)
        self.rawplot.setGeometry(QtCore.QRect(10, 8, 425, 507))
        self.rawplot.setObjectName("rawplot")
        self.frame = QtWidgets.QFrame(self.centralwidget)
        self.frame.setGeometry(QtCore.QRect(10, 10, 151, 257))
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.pushButton = QtWidgets.QPushButton(self.frame)
        self.pushButton.setGeometry(QtCore.QRect(20, 18, 111, 51))
        self.pushButton.setObjectName("pushButton")
        self.pushButton_2 = QtWidgets.QPushButton(self.frame)
        self.pushButton_2.setGeometry(QtCore.QRect(20, 78, 111, 51))
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_3 = QtWidgets.QPushButton(self.frame)
        self.pushButton_3.setGeometry(QtCore.QRect(20, 138, 111, 51))
        self.pushButton_3.setObjectName("pushButton_3")
        self.checkBox = QtWidgets.QCheckBox(self.frame)
        self.checkBox.setGeometry(QtCore.QRect(20, 198, 117, 41))
        font = QtGui.QFont()
        font.setFamily("DejaVu Sans")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.checkBox.setFont(font)
        self.checkBox.setObjectName("checkBox")
        self.frame_5 = QtWidgets.QFrame(self.centralwidget)
        self.frame_5.setGeometry(QtCore.QRect(466, 308, 467, 521))
        self.frame_5.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_5.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_5.setObjectName("frame_5")
        self.rawplot_2 = ImageView(self.frame_5)
        self.rawplot_2.setGeometry(QtCore.QRect(10, 10, 453, 505))
        self.rawplot_2.setObjectName("rawplot_2")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(174, 38, 121, 73))
        font = QtGui.QFont()
        font.setFamily("DejaVu Sans")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_2.setFont(font)
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(186, 150, 99, 105))
        font = QtGui.QFont()
        font.setFamily("DejaVu Sans")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_3.setFont(font)
        self.label_3.setAlignment(QtCore.Qt.AlignCenter)
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(22, 286, 131, 21))
        font = QtGui.QFont()
        font.setFamily("DejaVu Sans")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_4.setFont(font)
        self.label_4.setObjectName("label_4")
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(468, 286, 201, 21))
        font = QtGui.QFont()
        font.setFamily("DejaVu Sans")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_5.setFont(font)
        self.label_5.setObjectName("label_5")
        self.frame_6 = QtWidgets.QFrame(self.centralwidget)
        self.frame_6.setGeometry(QtCore.QRect(302, 22, 637, 111))
        self.frame_6.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_6.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_6.setObjectName("frame_6")
        self.grplot = PlotWidget(self.frame_6)
        self.grplot.setGeometry(QtCore.QRect(6, 6, 625, 101))
        self.grplot.setObjectName("grplot")
        self.frame_8 = QtWidgets.QFrame(self.centralwidget)
        self.frame_8.setGeometry(QtCore.QRect(298, 152, 639, 113))
        self.frame_8.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_8.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_8.setObjectName("frame_8")
        self.grplot_2 = PlotWidget(self.frame_8)
        self.grplot_2.setGeometry(QtCore.QRect(6, 6, 627, 101))
        self.grplot_2.setObjectName("grplot_2")
        self.label_6 = QtWidgets.QLabel(self.centralwidget)
        self.label_6.setGeometry(QtCore.QRect(948, -6, 129, 57))
        font = QtGui.QFont()
        font.setFamily("DejaVu Sans")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_6.setFont(font)
        self.label_6.setObjectName("label_6")
        self.dir_icon = QtWidgets.QLabel(self.centralwidget)
        self.dir_icon.setGeometry(QtCore.QRect(184, 106, 67, 65))
        self.dir_icon.setText("")
        self.dir_icon.setObjectName("dir_icon")
        self.label_7 = QtWidgets.QLabel(self.centralwidget)
        self.label_7.setGeometry(QtCore.QRect(938, 392, 161, 77))
        font = QtGui.QFont()
        font.setFamily("DejaVu Sans")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_7.setFont(font)
        self.label_7.setObjectName("label_7")
        self.frame_13 = QtWidgets.QFrame(self.centralwidget)
        self.frame_13.setGeometry(QtCore.QRect(1052, 16, 571, 93))
        self.frame_13.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_13.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_13.setObjectName("frame_13")
        self.pop_dir = PlotWidget(self.frame_13)
        self.pop_dir.setGeometry(QtCore.QRect(6, 6, 561, 83))
        self.pop_dir.setObjectName("pop_dir")
        self.frame_14 = QtWidgets.QFrame(self.centralwidget)
        self.frame_14.setGeometry(QtCore.QRect(1052, 114, 571, 93))
        self.frame_14.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_14.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_14.setObjectName("frame_14")
        self.pop_vel = PlotWidget(self.frame_14)
        self.pop_vel.setGeometry(QtCore.QRect(6, 6, 561, 83))
        self.pop_vel.setObjectName("pop_vel")
        self.frame_15 = QtWidgets.QFrame(self.centralwidget)
        self.frame_15.setGeometry(QtCore.QRect(1052, 212, 571, 93))
        self.frame_15.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_15.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_15.setObjectName("frame_15")
        self.pop_freq = PlotWidget(self.frame_15)
        self.pop_freq.setGeometry(QtCore.QRect(6, 6, 561, 83))
        self.pop_freq.setObjectName("pop_freq")
        self.frame_16 = QtWidgets.QFrame(self.centralwidget)
        self.frame_16.setGeometry(QtCore.QRect(1052, 310, 571, 93))
        self.frame_16.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_16.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_16.setObjectName("frame_16")
        self.pop_con = PlotWidget(self.frame_16)
        self.pop_con.setGeometry(QtCore.QRect(6, 6, 561, 83))
        self.pop_con.setObjectName("pop_con")
        self.frame_17 = QtWidgets.QFrame(self.centralwidget)
        self.frame_17.setGeometry(QtCore.QRect(1052, 434, 571, 93))
        self.frame_17.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_17.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_17.setObjectName("frame_17")
        self.single_dir = PlotWidget(self.frame_17)
        self.single_dir.setGeometry(QtCore.QRect(6, 6, 561, 83))
        self.single_dir.setObjectName("single_dir")
        self.frame_18 = QtWidgets.QFrame(self.centralwidget)
        self.frame_18.setGeometry(QtCore.QRect(1052, 534, 571, 93))
        self.frame_18.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_18.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_18.setObjectName("frame_18")
        self.single_vel = PlotWidget(self.frame_18)
        self.single_vel.setGeometry(QtCore.QRect(6, 6, 561, 83))
        self.single_vel.setObjectName("single_vel")
        self.frame_19 = QtWidgets.QFrame(self.centralwidget)
        self.frame_19.setGeometry(QtCore.QRect(1052, 634, 571, 93))
        self.frame_19.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_19.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_19.setObjectName("frame_19")
        self.single_freq = PlotWidget(self.frame_19)
        self.single_freq.setGeometry(QtCore.QRect(6, 6, 561, 83))
        self.single_freq.setObjectName("single_freq")
        self.frame_20 = QtWidgets.QFrame(self.centralwidget)
        self.frame_20.setGeometry(QtCore.QRect(1052, 734, 571, 93))
        self.frame_20.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_20.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_20.setObjectName("frame_20")
        self.single_con = PlotWidget(self.frame_20)
        self.single_con.setGeometry(QtCore.QRect(6, 6, 561, 83))
        self.single_con.setObjectName("single_con")
        self.label_8 = QtWidgets.QLabel(self.centralwidget)
        self.label_8.setGeometry(QtCore.QRect(960, 40, 93, 57))
        font = QtGui.QFont()
        font.setFamily("DejaVu Sans")
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.label_8.setFont(font)
        self.label_8.setObjectName("label_8")
        self.label_9 = QtWidgets.QLabel(self.centralwidget)
        self.label_9.setGeometry(QtCore.QRect(960, 130, 93, 57))
        font = QtGui.QFont()
        font.setFamily("DejaVu Sans")
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.label_9.setFont(font)
        self.label_9.setObjectName("label_9")
        self.label_10 = QtWidgets.QLabel(self.centralwidget)
        self.label_10.setGeometry(QtCore.QRect(956, 230, 93, 57))
        font = QtGui.QFont()
        font.setFamily("DejaVu Sans")
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.label_10.setFont(font)
        self.label_10.setObjectName("label_10")
        self.label_11 = QtWidgets.QLabel(self.centralwidget)
        self.label_11.setGeometry(QtCore.QRect(956, 330, 93, 57))
        font = QtGui.QFont()
        font.setFamily("DejaVu Sans")
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.label_11.setFont(font)
        self.label_11.setObjectName("label_11")
        self.label_12 = QtWidgets.QLabel(self.centralwidget)
        self.label_12.setGeometry(QtCore.QRect(958, 452, 93, 57))
        font = QtGui.QFont()
        font.setFamily("DejaVu Sans")
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.label_12.setFont(font)
        self.label_12.setObjectName("label_12")
        self.label_13 = QtWidgets.QLabel(self.centralwidget)
        self.label_13.setGeometry(QtCore.QRect(960, 550, 93, 57))
        font = QtGui.QFont()
        font.setFamily("DejaVu Sans")
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.label_13.setFont(font)
        self.label_13.setObjectName("label_13")
        self.label_14 = QtWidgets.QLabel(self.centralwidget)
        self.label_14.setGeometry(QtCore.QRect(956, 650, 93, 57))
        font = QtGui.QFont()
        font.setFamily("DejaVu Sans")
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.label_14.setFont(font)
        self.label_14.setObjectName("label_14")
        self.label_15 = QtWidgets.QLabel(self.centralwidget)
        self.label_15.setGeometry(QtCore.QRect(956, 754, 93, 57))
        font = QtGui.QFont()
        font.setFamily("DejaVu Sans")
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.label_15.setFont(font)
        self.label_15.setObjectName("label_15")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1633, 22))
        self.menubar.setObjectName("menubar")
        self.menuRASP_Display = QtWidgets.QMenu(self.menubar)
        self.menuRASP_Display.setObjectName("menuRASP_Display")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.menubar.addAction(self.menuRASP_Display.menuAction())

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
        self.label_2.setText(_translate("MainWindow", "Population\n"
"Average"))
        self.label_3.setText(_translate("MainWindow", "Selected\n"
"Neuron"))
        self.label_4.setText(_translate("MainWindow", "Raw Frame"))
        self.label_5.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" color:#eeeeec;\">Processed Frame</span></p></body></html>"))
        self.label_6.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" color:#eeeeec;\">Population</span><br><span style=\" color:#eeeeec;\">tuning</span></p></body></html>"))
        self.label_7.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" color:#eeeeec;\">Selected neuron<br/>tuning</span></p></body></html>"))
        self.label_8.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:12pt; color:#eeeeec;\">Direction</span></p></body></html>"))
        self.label_9.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:12pt; color:#eeeeec;\">Speed</span></p></body></html>"))
        self.label_10.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:12pt; color:#eeeeec;\">Spatial<br/>frequency</span></p></body></html>"))
        self.label_11.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:12pt; color:#eeeeec;\">Contrast</span></p></body></html>"))
        self.label_12.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:12pt; color:#eeeeec;\">Direction</span></p></body></html>"))
        self.label_13.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:12pt; color:#eeeeec;\">Speed</span></p></body></html>"))
        self.label_14.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:12pt; color:#eeeeec;\">Spatial<br/>frequency</span></p></body></html>"))
        self.label_15.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:12pt; color:#eeeeec;\">Contrast</span></p></body></html>"))
        self.menuRASP_Display.setTitle(_translate("MainWindow", "Nexus Display"))


from pyqtgraph import ImageView, PlotWidget
