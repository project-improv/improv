from PyQt5 import QtCore, QtGui, QtWidgets
from pyqtgraph import ImageView, PlotWidget

class rawplot:

    frameList = []
    labelList = []
    framelistbareraw= []
    labellistbare = []

    def makebottomleft(self, MainWindow):
        self.frame_2 = QtWidgets.QFrame(self.centralwidget)
        self.frameList.append(self.frame_2) 
        self.framelistbareraw.append("frame_2")
        self.frame_2.setGeometry(QtCore.QRect(20, 280, 331, 321))
        
        self.rawplot = ImageView(self.frame_2)
        self.frameList.append(self.rawplot)
        self.framelistbareraw.append("rawplot")
        self.rawplot.setGeometry(QtCore.QRect(10, 10, 321, 311))

        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.labelList.append(self.label_4)
        self.labellistbare.append("label_4")
        self.label_4.setGeometry(QtCore.QRect(20, 260, 131, 21))


    def makebottomcenter(self, MainWindow):
        self.frame_5 = QtWidgets.QFrame(self.centralwidget)
        self.frame_5.setGeometry(QtCore.QRect(390, 280, 341, 321))
        self.frameList.append(self.frame_5)
        self.framelistbareraw.append("frame_5")

        self.rawplot_2 = ImageView(self.frame_5)
        self.rawplot_2.setGeometry(QtCore.QRect(10, 10, 331, 311))
        self.frameList.append(self.rawplot_2)
        self.framelistbareraw.append("rawplot_2")

        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.labelList.append(self.label_5)
        self.labellistbare.append("label_5")
        self.label_5.setGeometry(QtCore.QRect(390, 260, 131, 21))


    def makebottomright(self):
        self.frame_4 = QtWidgets.QFrame(self.centralwidget)
        self.frame_4.setGeometry(QtCore.QRect(760, 280, 341, 321))
        self.frameList.append(self.frame_4) 
        self.framelistbareraw.append("frame_4")

        self.rawplot_3 = ImageView(self.frame_4)
        self.frameList.append(self.rawplot_3) 
        self.framelistbareraw.append("rawplot_3")
        self.rawplot_3.setGeometry(QtCore.QRect(10, 10, 331, 311))

    def createrawplots(self, MainWindow, type):

        if 0 in type:
            self.makebottomleft(self)
        if 1 in type: 
            self.makebottomcenter(self)
        if 2 in type:
            self.makebottomright()

        for x in range(len(type)):
            print(x)
            self.frameList[2*x].setFrameShape(QtWidgets.QFrame.StyledPanel)
            self.frameList[2*x].setFrameShadow(QtWidgets.QFrame.Raised)
            self.frameList[2*x].setObjectName(self.framelistbareraw[2*x])
            self.frameList[(2*x)+1].setObjectName(self.framelistbareraw[(2*x)+1])

            if type[x] != 2:
                font = QtGui.QFont()
                font.setFamily("Helvetica Neue")
                font.setPointSize(14)
                font.setBold(True)
                font.setWeight(75)
                if type[x] == 0:
                    self.label_4.setFont(font)
                    self.label_4.setObjectName("label_4")

                if type[x] == 1:
                    self.label_5.setFont(font)
                    self.label_5.setObjectName("label_5")


        if 0 in type:
            _translate = QtCore.QCoreApplication.translate
            self.label_4.setText(_translate("MainWindow", "Raw Plot"))
            QtCore.QMetaObject.connectSlotsByName(MainWindow)
            self.rawplot.ui.histogram.vb.setLimits(yMin=-0.1, yMax=200)

        if 1 in type:
            _translate = QtCore.QCoreApplication.translate
            self.label_5.setText(_translate("MainWindow", "Processed Frame"))
            QtCore.QMetaObject.connectSlotsByName(MainWindow)

            

            