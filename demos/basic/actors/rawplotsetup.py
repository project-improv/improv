from PyQt5 import QtCore, QtGui, QtWidgets
from pyqtgraph import ImageView, PlotWidget

class rawplot:
    def makebottomright(self, MainWindow):
        self.frame_2 = QtWidgets.QFrame(self.centralwidget) 
        self.frame_2.setGeometry(QtCore.QRect(20, 280, 331, 321))
        self.frame_2.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_2.setObjectName("frame_2")
        self.rawplot = ImageView(self.frame_2)
        self.rawplot.setGeometry(QtCore.QRect(10, 10, 321, 311))
        self.rawplot.setObjectName("rawplot")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(20, 260, 131, 21))
        font = QtGui.QFont()
        font.setFamily("Helvetica Neue")
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.label_4.setFont(font)
        self.label_4.setObjectName("label_4")

        _translate = QtCore.QCoreApplication.translate
        self.label_4.setText(_translate("MainWindow", "Raw Plot"))
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def makebottomcenter(self, MainWindow):
        self.frame_5 = QtWidgets.QFrame(self.centralwidget)
        self.frame_5.setGeometry(QtCore.QRect(390, 280, 341, 321))
        self.frame_5.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_5.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_5.setObjectName("frame_5")
        self.rawplot_2 = ImageView(self.frame_5)
        self.rawplot_2.setGeometry(QtCore.QRect(10, 10, 331, 311))
        self.rawplot_2.setObjectName("rawplot_2")
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(390, 260, 131, 21))
        font = QtGui.QFont()
        font.setFamily("Helvetica Neue")
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.label_5.setFont(font)
        self.label_5.setObjectName("label_5")
        _translate = QtCore.QCoreApplication.translate
        self.label_5.setText(_translate("MainWindow", "Processed Frame"))
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    
    def makebottomleft(self):
        self.frame_4 = QtWidgets.QFrame(self.centralwidget)
        self.frame_4.setGeometry(QtCore.QRect(760, 280, 341, 321))
        self.frame_4.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_4.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_4.setObjectName("frame_4")
        self.rawplot_3 = ImageView(self.frame_4)
        self.rawplot_3.setGeometry(QtCore.QRect(10, 10, 331, 311))
        self.rawplot_3.setObjectName("rawplot_3")
