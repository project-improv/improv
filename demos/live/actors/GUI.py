from PyQt5 import QtGui,QtCore,QtWidgets
from PyQt5.QtGui import QColor, QPixmap
from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtWidgets import QMessageBox, QFileDialog
from . import improv_viz_stim_GP as improv_viz_stim
from improv.store import Limbo
from improv.actor import Spike
import numpy as np
from math import floor
import time
import pyqtgraph
from pyqtgraph import EllipseROI, PolyLineROI, ColorMap, ROI, LineSegmentROI
from queue import Empty
from matplotlib import cm
from matplotlib.colors import ListedColormap
from demos.pandas.Pandas3D import live_QPanda3D
from QPanda3D.QPanda3DWidget import QPanda3DWidget
from PyQt5.QtWidgets import QGridLayout

import logging; logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

#NOTE: GUI only gives comm signals to Nexus, does not receive any. Visual serves that role
#TODO: Add ability to receive signals like pause updating ...?

class FrontEnd(QtWidgets.QMainWindow, improv_viz_stim.Ui_MainWindow):

    COLOR = {0: ( 240, 122,  5),
             1: (181, 240,  5),
             2: (5, 240,  5),
             3: (5,  240,  181),
             4: (5,  122, 240),
             5: (64,  5, 240),
             6: ( 240,  5, 240),
             7: ( 240, 5, 64)}

    def __init__(self, visual, comm, parent=None):
        ''' Setup GUI
            Setup and start Nexus controls
        '''
        self.visual = visual #Visual class that provides plots and images
        self.comm = comm #Link back to Nexus for transmitting signals

        self.total_times = []
        self.first = True
        self.prev = 0

        pyqtgraph.setConfigOption('background', QColor(100, 100, 100))
        super(FrontEnd, self).__init__(parent)
        self.setupUi(self)
        pyqtgraph.setConfigOptions(leftButtonPan=False)

        self.customizePlots()

        self.pushButton_3.clicked.connect(_call(self._runProcess)) #Tell Nexus to start
        self.pushButton_3.clicked.connect(_call(self.update)) #Update front-end graphics
        self.pushButton_2.clicked.connect(_call(self._setup))
        self.pushButton.clicked.connect(_call(self._loadParams)) #File Dialog, then tell Nexus to load tweak
        self.checkBox.stateChanged.connect(self.update) #Show live front-end updates
        
        # self.rawplot_2.getImageItem().mouseClickEvent = self.mouseClick #Select a neuron

        self.xs = {}
        # self.xs['angle'] = np.linspace(0,350,num=15)
        # self.xs['vel'] = np.array([0.02, 0.06, 0.1]) #np.around(np.linspace(0.02, 0.1, num=4), decimals=2)
        # self.xs['freq'] = np.array([20, 40, 60]) #np.arange(5,80,5)
        # self.xs['contrast'] = np.arange(5)
        self.xs['angle'] = self.visual.stimuli[0]
        self.xs['vel'] = self.visual.stimuli[1]
        # self.xs['freq'] = self.visual.stimuli[2]
        # self.xs['contrast'] = self.visual.stimuli[3]

    def update(self):
        ''' Update visualization while running
        '''
        t = time.time()
        #start looking for data to display
        self.visual.getData()
        #logger.info('Did I get something:', self.visual.Cx)
        if self.draw:
            #plot lines
            try:
                self.updateLines()
            except Exception as e:
                print('update lines error {}'.format(e))
                import traceback
                print('---------------------Exception in update lines: ' , traceback.format_exc())

            #plot video
            try:
                self.updateVideo()
            except Exception as e:
                logger.error('Error in FrontEnd update Video:  {}'.format(e))
                import traceback
                print('---------------------Exception in update video: ' , traceback.format_exc())
        #re-update
        if self.checkBox.isChecked():
            self.draw = True
        else:
            self.draw = False    
        self.visual.draw = self.draw
        self.loadPandas()
            
        QtCore.QTimer.singleShot(10, self.update)
        
        self.total_times.append([self.visual.frame_num, time.time()-t])

    def loadPandas(self):
        if(self.visual.flag):
            self.data = list(self.visual.idsStim.values())[0][2: len(list(self.visual.idsStim.values())[0])]
            self.data.pop(1)
            if(self.first):
                world = live_QPanda3D.PandaTest()
                world.get_size()
                world.createCard(self.data[0], self.data[1], self.data[2], self.data[3])
                pandaWidget = QPanda3DWidget(world)
                layout = QGridLayout()
                layout.addWidget(pandaWidget, 1, 0)
                self.pop_con.setLayout(layout)
                self.first = False
            else:
                messenger.send("stimulus", self.data)
            self.visual.flag = False

    def customizePlots(self):
        self.checkBox.setChecked(True)
        self.draw = True

        pixmap = QPixmap('src/visual/rainbow_dir.png')
        pixmap = pixmap.scaled(60, 60) #, QtCore.Qt.KeepAspectRatio)
        self.dir_icon.setPixmap(pixmap)

        #init line plot
        self.flag = True
        self.flagW = True
        self.flagL = True
        self.last_x = None
        self.last_y = None
        self.weightN = None
        self.last_n = None

        self.c1 = self.grplot.plot(clipToView=True)
        self.c1_stim = [self.grplot.plot(clipToView=True) for _ in range(len(self.COLOR))]
        self.c2 = self.grplot_2.plot()
        grplot = [self.grplot, self.grplot_2]
        for plt in grplot:
            plt.getAxis('bottom').setTickSpacing(major=50, minor=50)
        #    plt.setLabel('bottom', "Frames")
        #    plt.setLabel('left', "Temporal traces")
        self.updateLines()
        self.activePlot = 'r'

        # self.pop_lines = {}
        # self.pop_lines['angle'] = self.pop_dir.plot()
        # self.pop_lines['vel'] = self.pop_vel.plot()
        # self.pop_lines['freq'] = self.pop_freq.plot()
        # self.pop_lines['contrast'] = self.pop_con.plot()

        # self.single_lines = {}
        # self.single_lines['angle'] = self.single_dir.plot()
        # self.single_lines['vel'] = self.single_vel.plot()
        # self.single_lines['freq'] = self.single_freq.plot()
        # self.single_lines['contrast'] = self.single_con.plot()

        # self.quant = self.single_dir.plot()
        # self.GPest = self.single_vel.plot()
        # self.GPuncert = self.single_freq.plot()

        #videos
        # self.rawplot.ui.histogram.vb.disableAutoRange()
        self.rawplot.ui.histogram.vb.setLimits(yMin=-0.1, yMax=200) #0-255 needed, saturated here for easy viewing


        # if self.visual.showConnectivity:
        #     self.rawplot_3.setColorMap(cmapToColormap(cm.inferno))
        self.GP_est.ui.histogram.hide()
        self.GP_est.ui.roiBtn.hide()
        self.GP_est.ui.menuBtn.hide()
        self.GP_est.setColorMap(cmapToColormap(cm.inferno))

        self.GP_unc.ui.histogram.hide()
        self.GP_unc.ui.roiBtn.hide()
        self.GP_unc.ui.menuBtn.hide()
        self.GP_unc.setColorMap(cmapToColormap(cm.inferno))

        self.quant_tc.ui.histogram.hide()
        self.quant_tc.ui.roiBtn.hide()
        self.quant_tc.ui.menuBtn.hide()

    def _loadParams(self):
        ''' Button event to load parameters from file
            File location determined from user input
            Throws FileNotFound error
        '''
        fname = QFileDialog.getOpenFileName(self, 'Open file', 'demos/')
            #TODO: make default home folder system-independent
        try:
            self._loadTweak(fname[0])
        except FileNotFoundError as e:
            logger.error('File not found {}'.format(e))
            #raise FileNotFoundError
    
    def _runProcess(self):
        '''Run ImageProcessor in separate thread
        '''
        self.comm.put([Spike.run()])
        logger.info('-------------------------   put run in comm')
        #TODO: grey out button until self.t is done, but allow other buttons to be active

    def _setup(self):
        self.comm.put([Spike.setup()])
        self.visual.setup()

    def _loadTweak(self, file):
        self.comm.put(['load', file])
    
    def updateVideo(self):
        ''' TODO: Bug on clicking ROI --> trace and report to pyqtgraph
        '''
        image = None
        raw, color, est, unc = self.visual.getFrames()
        if raw is not None:
            raw = np.rot90(raw,2)
            if np.unique(raw).size > 1:
                self.rawplot.setImage(raw) #, autoHistogramRange=False)
                self.rawplot.ui.histogram.vb.setLimits(yMin=80, yMax=200)
        if color is not None:
            color = np.rot90(color,2)
            self.rawplot_2.setImage(color)
            # self.rawplot_2.ui.histogram.vb.setLimits(yMin=8, yMax=255)
        if est is not None:
            self.GP_est.setImage(est)
        if unc is not None:
            self.GP_unc.setImage(unc)

    def updateLines(self):
        ''' Helper function to plot the line traces
            of the activity of the selected neurons.
            TODO: separate updates for each plot?
        '''

        ## Add current stimulus information: orientation change, motion starting, etc. Better timing locked

        penW=pyqtgraph.mkPen(width=2, color='w')
        penR=pyqtgraph.mkPen(width=2, color='r')
        penG=pyqtgraph.mkPen(width=3, color='g')
        C = None
        Cx = None
        tune = None
        y_results = None
        stimText = None
        try:
            (Cx, C, Cpop, tune, y_results, stimText) = self.visual.getCurves()
        except TypeError:
            pass
        except Exception as e:
            logger.error('Output does not likely exist. Error: {}'.format(e))

        if stimText:
            # print(stimText)
            texthere = ','.join(map(str,stimText[0][2:]) )
            # print('sewtting GUI label', texthere)
            self.label_10.setText(texthere)
    
        try:
            self.label_7.setText('Selected neuron: ', self.visual.selectedNeuron)
        except: pass

        if (C is not None and Cx is not None):
            self.c1.setData(Cx, Cpop, pen=penW)

            for i, plot in enumerate(self.c1_stim):
                # print(self.visual.allStims)
                try:
                    if len(self.visual.allStims[i]) > 0:
                        # display = np.array(self.visual.allStims[i])
                        d = []
                        for s in self.visual.allStims[i]:
                            d.extend(np.arange(s,s+10).tolist())
                        # display = np.arange(self.visual.allStims[i], self.visual.allStims[i]+15)
                        display = d
                        display = np.clip(display, np.min(Cx), np.max(Cx))
                        try:
                            plot.setData(display, [int(np.max(Cpop))+1] * len(display),
                                    symbol='s', symbolSize=6, antialias=False,
                                    pen=None, symbolPen=self.COLOR[i], symbolBrush=self.COLOR[i])
                        except:
                            print(display)
                except KeyError:
                    pass

            self.c2.setData(Cx, C, pen=penR)
            
            # if(self.flag or self.visual.flagN):
            #     self.selected = self.visual.selectedNeuron
            #     self._updateRedCirc()
            #     self.visual.flagN = False
                # self.selected = self.visual.getFirstSelect()
                # if self.selected is not None:
                #     self._updateRedCirc()

        # if y_results is not None:
        #     for key in y_results.keys():
        #         self.single_lines[key].setData(self.xs[key], y_results[key][self.visual.selectedNeuron])
                # pop_val = np.nanmean(y_results[key], axis=0)
                # self.pop_lines[key].setData(self.xs[key], pop_val)
        
    def mouseClick(self, event):
        '''Clicked on processed image to select neurons
        '''
        #TODO: make this unclickable until finished updated plot (?)
        event.accept()
        mousePoint = event.pos()
        self.selected = self.visual.selectNeurons(int(mousePoint.x()), int(mousePoint.y()))
        selectedraw = np.zeros(2)
        selectedraw[0] = int(mousePoint.x())
        selectedraw[1] = int(mousePoint.y())
        self._updateRedCirc()

        if self.last_n is None:
            self.last_n = self.visual.selectedNeuron

        # if self.flagW: #nothing drawn yet
        #     loc, lines, strengths = self.visual.selectNW(selectedraw[0], selectedraw[1])
        #     # print('clicked lines ', lines)
        #     self.lines = []
        #     self.pens = []
        #     colors =['g']*9 + ['r']*9
        #     for i in range(18):
        #         n = lines[i]
        #         if strengths[i] > 1e-6:
        #             if strengths[i] > 1e-4:
        #                 self.pens.append(pyqtgraph.mkPen(width=2, color=colors[i]))
        #             else:
        #                 self.pens.append(pyqtgraph.mkPen(width=1, color=colors[i]))
        #             self.lines.append(LineSegmentROI(positions=([n[0],n[2]],[n[1],n[3]]), handles=(None,None), pen=self.pens[i], movable=False))
        #             self.rawplot_2.getView().addItem(self.lines[i])
        #         else:
        #             self.pens.append(pyqtgraph.mkPen(width=1, color=colors[i]))
        #             self.lines.append(LineSegmentROI(positions=([n[0],n[0]],[n[0],n[0]]), handles=(None,None), pen=self.pens[i], movable=False))
        #             self.rawplot_2.getView().addItem(self.lines[i])

        #     self.last_n = self.visual.selectedNeuron
        #     self.flagW = False
        elif self.last_n == self.visual.selectedNeuron:
            for i in range(18):
                self.rawplot_2.getView().removeItem(self.lines[i])
            self.flagW = True

    def _updateRedCirc(self):
        ''' Circle neuron whose activity is in top (red) graph
            Default is neuron #0 from initialize
            #TODO: add arg instead of self.selected
        '''
        ROIpen1=pyqtgraph.mkPen(width=1, color='r')
        if self.flag:
            self.red_circ = CircleROI(pos = np.array([self.selected[0][0], self.selected[0][1]])-5, size=10, movable=False, pen=ROIpen1)
            self.rawplot_2.getView().addItem(self.red_circ)
            self.flag = False
        if np.count_nonzero(self.selected[0]) > 0:
            self.rawplot_2.getView().removeItem(self.red_circ)
            self.red_circ = CircleROI(pos = np.array([self.selected[0][0], self.selected[0][1]])-5, size=10, movable=False, pen=ROIpen1)
            self.rawplot_2.getView().addItem(self.red_circ)

    def closeEvent(self, event):
        '''Clicked x/close on window
            Add confirmation for closing without saving
        '''
        confirm = QMessageBox.question(self, 'Message', 'Quit without saving?',
                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if confirm == QMessageBox.Yes:
            self.comm.put(['quit'])
            # print('Visual broke, avg time per frame: ', np.mean(self.visual.total_times, axis=0))
            print('Visual got through ', self.visual.frame_num, ' frames')
            # print('GUI avg time ', np.mean(self.total_times))
            np.savetxt('output/timing/visual_frame_time.txt', np.array(self.visual.total_times))
            np.savetxt('output/timing/gui_frame_time.txt', np.array(self.total_times))
            np.savetxt('output/timing/visual_timestamp.txt', np.array(self.visual.timestamp))
            event.accept()
        else: event.ignore()

def _call(fnc, *args, **kwargs):
    ''' Call handler for (external) events
    '''
    def _callback():
        return fnc(*args, **kwargs)
    return _callback

class CircleROI(EllipseROI):
    def __init__(self, pos, size, **args):
        pyqtgraph.ROI.__init__(self, pos, size, **args)
        self.aspectLocked = True

class PolyROI(PolyLineROI):
    def __init__(self, positions, pos, **args):
        closed = True
        print('got positions ', positions)
        pyqtgraph.ROI.__init__(self, positions, closed, pos, **args)

def cmapToColormap(cmap: ListedColormap) -> ColorMap:
    """ Converts matplotlib cmap to pyqtgraph ColorMap. """

    colordata = (np.array(cmap.colors) * 255).astype(np.uint8)
    indices = np.linspace(0., 1., len(colordata))
    return ColorMap(indices, colordata)


if __name__=="__main__":
    import sys
    app = QtGui.QApplication(sys.argv)
    rasp = FrontEnd(None,None)
    rasp.show()
    app.exec_()
