# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file './improv_basic.ui'
#
# Created by: PyQt5 UI code generator 5.12.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1118, 652)
        MainWindow.setStyleSheet("QMainWindow { background-color: 'blue'; }")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setStyleSheet(
            "#centralwidget { background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:0, y2:1, stop:0 rgb(50, 50, 50), stop:1 rgba(80, 80, 80, 255)); }\n"
            "#checkBox {color: black}\n"
            "#label_2 {color: rgb(225, 230, 240)}\n"
            "#label_3 {color: rgb(225, 230, 240)}\n"
            "#label_4 {color: rgb(225, 230, 240)}\n"
            "#label_5 {color: rgb(225, 230, 240)}\n"
            "#label_6 {color: rgb(225, 230, 240)}\n"
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
            "}"
        )
        self.centralwidget.setObjectName("centralwidget")
        self.frame_2 = QtWidgets.QFrame(self.centralwidget)
        self.frame_2.setGeometry(QtCore.QRect(20, 280, 331, 321))
        self.frame_2.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_2.setObjectName("frame_2")
        self.rawplot = ImageView(self.frame_2)
        self.rawplot.setGeometry(QtCore.QRect(10, 10, 321, 311))
        self.rawplot.setObjectName("rawplot")
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
        self.frame_4 = QtWidgets.QFrame(self.centralwidget)
        self.frame_4.setGeometry(QtCore.QRect(760, 280, 341, 321))
        self.frame_4.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_4.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_4.setObjectName("frame_4")
        self.rawplot_3 = ImageView(self.frame_4)
        self.rawplot_3.setGeometry(QtCore.QRect(10, 10, 331, 311))
        self.rawplot_3.setObjectName("rawplot_3")
        self.frame_5 = QtWidgets.QFrame(self.centralwidget)
        self.frame_5.setGeometry(QtCore.QRect(390, 280, 341, 321))
        self.frame_5.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_5.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_5.setObjectName("frame_5")
        self.rawplot_2 = ImageView(self.frame_5)
        self.rawplot_2.setGeometry(QtCore.QRect(10, 10, 331, 311))
        self.rawplot_2.setObjectName("rawplot_2")
        self.frame_3 = QtWidgets.QFrame(self.centralwidget)
        self.frame_3.setGeometry(QtCore.QRect(760, 20, 341, 241))
        self.frame_3.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_3.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_3.setObjectName("frame_3")
        self.slider = QtWidgets.QSlider(self.frame_3)
        self.slider.setGeometry(QtCore.QRect(30, 70, 131, 22))
        self.slider.setOrientation(QtCore.Qt.Horizontal)
        self.slider.setObjectName("slider")
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
        self.frame_10 = QtWidgets.QFrame(self.frame_3)
        self.frame_10.setGeometry(QtCore.QRect(194, 64, 133, 125))
        self.frame_10.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_10.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_10.setObjectName("frame_10")
        self.grplot_5 = PlotWidget(self.frame_10)
        self.grplot_5.setGeometry(QtCore.QRect(6, 6, 121, 113))
        self.grplot_5.setObjectName("grplot_5")
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
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(20, 260, 131, 21))
        font = QtGui.QFont()
        font.setFamily("Helvetica Neue")
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.label_4.setFont(font)
        self.label_4.setObjectName("label_4")
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(390, 260, 131, 21))
        font = QtGui.QFont()
        font.setFamily("Helvetica Neue")
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.label_5.setFont(font)
        self.label_5.setObjectName("label_5")
        self.label_6 = QtWidgets.QLabel(self.centralwidget)
        self.label_6.setGeometry(QtCore.QRect(20, 611, 331, 21))
        font = QtGui.QFont()
        font.setFamily("Helvetica Neue")
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.label_6.setFont(font)
        self.label_6.setObjectName("label_6")
        self.frame_6 = QtWidgets.QFrame(self.centralwidget)
        self.frame_6.setGeometry(QtCore.QRect(170, 30, 431, 101))
        self.frame_6.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_6.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_6.setObjectName("frame_6")
        self.grplot = PlotWidget(self.frame_6)
        self.grplot.setGeometry(QtCore.QRect(6, 6, 419, 89))
        self.grplot.setObjectName("grplot")
        self.frame_7 = QtWidgets.QFrame(self.centralwidget)
        self.frame_7.setGeometry(QtCore.QRect(618, 30, 107, 103))
        self.frame_7.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_7.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_7.setObjectName("frame_7")
        self.grplot_3 = PlotWidget(self.frame_7)
        self.grplot_3.setGeometry(QtCore.QRect(6, 6, 95, 89))
        self.grplot_3.setObjectName("grplot_3")
        self.frame_8 = QtWidgets.QFrame(self.centralwidget)
        self.frame_8.setGeometry(QtCore.QRect(170, 154, 431, 101))
        self.frame_8.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_8.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_8.setObjectName("frame_8")
        self.grplot_2 = PlotWidget(self.frame_8)
        self.grplot_2.setGeometry(QtCore.QRect(6, 6, 419, 89))
        self.grplot_2.setObjectName("grplot_2")
        self.frame_9 = QtWidgets.QFrame(self.centralwidget)
        self.frame_9.setGeometry(QtCore.QRect(619, 155, 105, 101))
        self.frame_9.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_9.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_9.setObjectName("frame_9")
        self.grplot_4 = PlotWidget(self.frame_9)
        self.grplot_4.setGeometry(QtCore.QRect(6, 6, 93, 89))
        self.grplot_4.setObjectName("grplot_4")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1118, 22))
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
        self.pushButton.setText(_translate("MainWindow", "Load\n" "Parameters"))
        self.pushButton_2.setText(_translate("MainWindow", "Setup"))
        self.pushButton_3.setText(_translate("MainWindow", "Run"))
        self.checkBox.setText(_translate("MainWindow", " Live Update"))
        self.label.setText(_translate("MainWindow", "Targeting Selection"))
        self.pushButton_4.setText(_translate("MainWindow", "Stimulate"))
        self.label_6.setText(_translate("MainWindow", "Threshold"))
        self.label_7.setText(_translate("MainWindow", "Direction"))
        self.label_2.setText(_translate("MainWindow", "Population Average"))
        self.label_3.setText(_translate("MainWindow", "Selected Neuron"))
        self.label_4.setText(_translate("MainWindow", "Raw Frame"))
        self.label_5.setText(_translate("MainWindow", "Processed Frame"))
        self.label_6.setText(_translate("MainWindow", "Number of Neurons: "))
        self.menuRASP_Display.setTitle(_translate("MainWindow", "Nexus Display"))


from pyqtgraph import ImageView, PlotWidget


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
