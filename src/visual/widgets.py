from PyQt5 import QtCore, QtGui, QtWidgets


class PathSlider(QtWidgets.QAbstractSlider):

    startChanged = QtCore.pyqtSignal(int, name='startChanged')
    endChanged = QtCore.pyqtSignal(int, name='endChanged')
    rangeChanged = QtCore.pyqtSignal(int, int, name='rangeChanged')  # self.startPoint, self.endPoint
    threshChanged = QtCore.pyqtSignal(int, name='threshChanged')

    def __init__(self, *args, minSize=150, path=None, **kwargs):
        """
        PyQt5 Widget: Path-based range slider (designed for circle)

        Dragging at any end adjusts the range.
        Dragging within the range moves the range.
        Dragging outside the range has no effects.

        :param minSize: Radius of the circle
        :type minSize: int
        :param path: QPainterPath of the intended shape
        :type path: QtGui.QPainterPath
        """
        super().__init__(*args, **kwargs)

        if path is None:  # Default to circle
            path = QtGui.QPainterPath(QtCore.QPointF(0, 0))
            path.addEllipse(0, 0, minSize, minSize)
        self.setPath(path)

        self._path = path
        self.strokePath = path
        self.scaledPath = path
        self.minSize = minSize

        self.border = 10
        self.setMinimumSize(minSize + 2*self.border + 30, 2*self.border + minSize)

        self.maxValue = 360
        self.startPoint = 0
        self.endPoint = 90
        self.sliderVal = 1

        self.dragging = 'start'  # or 'end'

        self.initLabels(minSize=minSize)

    def initLabels(self, minSize):
        """
        Initialize accessory labels and threshold sliders.

        """
        # Set Size
        fontSize = int(0.1 * minSize)
        sliderGap = int(0.2 * minSize)
        sliderMaxValue = 10
        sliderMinValue = 0

        # Set font
        self.font = QtGui.QFont()
        self.font.setPointSize(fontSize)

        # Define
        self.labelStartEnd = QtWidgets.QLabel(f'[{self.startPoint}°, {self.endPoint}°]', self)
        self.labelStartEnd.setAlignment(QtCore.Qt.AlignCenter)
        self.labelStartEnd.setFont(self.font)

        self.labelSize = QtWidgets.QLabel(str(sliderMinValue), self)
        self.labelSize.setAlignment(QtCore.Qt.AlignCenter)
        self.labelSize.setFont(self.font)

        self.labelStartEnd.setMinimumWidth(6 * fontSize)
        self.labelSize.setMinimumWidth(int(1.5 * fontSize))

        self.sliderThresh = QtWidgets.QSlider(self)
        self.sliderThresh.setMinimumHeight(minSize - sliderGap)
        self.sliderThresh.setMaximum(sliderMaxValue)
        self.sliderThresh.setMinimum(sliderMinValue)

        # Move
        self.labelStartEnd.move((minSize - fontSize//2) // 2 - int(1.8 * fontSize), minSize // 2 + 1)
        self.labelSize.move(minSize + sliderGap, 10)
        self.sliderThresh.move(minSize + sliderGap, sliderGap + 10)

        # Functionalize
        self.startChanged.connect(lambda x: self.labelStartEnd.setText(f'[{x}°, {self.endPoint}°]'))
        self.endChanged.connect(lambda x: self.labelStartEnd.setText(f'[{self.startPoint}°, {x}°]'))
        self.sliderThresh.valueChanged.connect(lambda x: self.labelSize.setText(str(x)))
        self.sliderThresh.sliderReleased.connect(lambda: self.threshChanged.emit(self.sliderThresh.value()))
        self.sliderThresh.sliderReleased.connect(lambda: setattr(self, 'sliderVal', self.sliderThresh.value()))

    def path(self):
        return self._path

    def setPath(self, path):
        path.translate(-path.boundingRect().topLeft())
        self._path = path
        self.update()

    path = QtCore.pyqtProperty(QtGui.QPainterPath, fget=path, fset=setPath)

    def paintEvent(self, event):
        # Define where the circle is
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)

        # Rotate so that 360 is at the top.
        tr = QtGui.QTransform()
        tr.rotate(-90)
        tr.translate(-self.minSize, 0)
        rotatedPath: QtGui.QPainterPath = tr.map(self.path)

        # Move and scale to fit window.
        # Need to do separately to reset coordinate system.
        tr = QtGui.QTransform()
        tr.translate(self.border, self.border)
        # scale = min((self.rect().width() - 2 * self.border) / self.path.boundingRect().width(),
        #             (self.rect().height() - 2 * self.border) / self.path.boundingRect().height())
        # tr.scale(scale, scale)
        self.scaledPath: QtGui.QPainterPath = tr.map(rotatedPath)

        # Generate fillable outline.
        stroker = QtGui.QPainterPathStroker()
        stroker.setCapStyle(QtCore.Qt.RoundCap)
        stroker.setWidth(8)

        # White contour
        stroke_path = stroker.createStroke(self.scaledPath).simplified()
        painter.setPen(QtGui.QPen(self.palette().color(QtGui.QPalette.Shadow), 1))
        painter.setBrush(QtGui.QBrush(self.palette().color(QtGui.QPalette.Midlight)))
        painter.drawPath(stroke_path)

        # Define highlight path
        stroker.setWidth(40)
        self.strokePath = stroker.createStroke(self.scaledPath).simplified()
        highlight_path = QtGui.QPainterPath()
        highlight_path.moveTo(self.scaledPath.pointAtPercent(self.startPoint / self.maxValue))  # Starting point

        angle_range = self.endPoint - self.startPoint
        end_angle = self.endPoint if angle_range > 0 else self.endPoint + self.maxValue

        for i in range(self.startPoint, end_angle):  # Tessellate, draw a line for every angle
            p = self.scaledPath.pointAtPercent(i % self.maxValue / self.maxValue)
            highlight_path.lineTo(p)

        # Draw highlight path
        stroker.setWidth(7)
        activeHighlight = self.palette().color(QtGui.QPalette.Highlight)
        painter.setPen(activeHighlight)
        painter.setBrush(QtGui.QBrush(QtGui.QColor(activeHighlight)))
        painter.drawPath(stroker.createStroke(highlight_path).simplified())

        opt = QtWidgets.QStyleOptionSlider()
        r = self.style().subControlRect(QtWidgets.QStyle.CC_Slider, opt, QtWidgets.QStyle.SC_SliderHandle, self)
        pixmap = QtGui.QPixmap(r.width() + 2*2, r.height() + 2*2)
        pixmap.fill(QtCore.Qt.transparent)
        r = pixmap.rect().adjusted(2, 2, -2, -2)

        pixmap_painter = QtGui.QPainter(pixmap)
        pixmap_painter.setRenderHint(QtGui.QPainter.Antialiasing)
        pixmap_painter.setPen(QtGui.QPen(self.palette().color(QtGui.QPalette.Shadow), 2))
        pixmap_painter.setBrush(self.palette().color(QtGui.QPalette.Base))
        pixmap_painter.drawRoundedRect(r, 4, 4)
        pixmap_painter.end()
        r.moveCenter(p.toPoint())
        painter.drawPixmap(r, pixmap)

    def _coordToPos(self, point: QtCore.QPoint):
        """ Convert QPoint into slider value """

        n_p = int((self.maximum() - self.minimum()) / self.singleStep())
        ls = []
        for i in range(n_p):
            p = self.scaledPath.pointAtPercent(i * 1.0 / n_p)
            ls.append(QtCore.QLineF(point, p).length())
        j = ls.index(min(ls))
        val = int(j * (self.maximum() - self.minimum()) / n_p)
        return val

    def update_pos(self, event: QtGui.QMouseEvent):  # Triggered at mouse click
        point = event.pos()

        if self.strokePath.contains(point):
            theta = self._coordToPos(point)
            if event.button() != QtCore.Qt.NoButton:  # User is clicking something.
                distStart = min([abs(theta - self.startPoint), abs(self.startPoint - theta + self.maxValue)])
                distEnd = min([abs(theta - self.endPoint), abs(self.endPoint - theta + self.maxValue)])

                if min(distEnd, distStart) < 20:  # Adjust ends only.
                    self.dragging = 'start' if distStart < distEnd else 'end'
                elif self._checkRange(theta):  # If within range, adjust both ends.
                    self.dragging = theta
                else:
                    self.dragging = None

            else:  # Continuous drag
                if self.dragging in ['start', 'end']:
                    self._changeValue(self.dragging, theta)

                elif isinstance(self.dragging, int):
                    self._changeValue('start', self.startPoint + (theta - self.dragging))
                    self._changeValue('end', self.endPoint + (theta - self.dragging))
                    self.dragging = theta

            self.update()

    def _changeValue(self, point, new):
        assert point in ['start', 'end']
        if new < 0:
            new += 360
        elif new >= 360:
            new -= 360
        setattr(self, f'{point}Point', new)
        eval(f'self.{point}Changed.emit(new)')

    def _checkRange(self, i):
        """ Check if {i} is within {self.startPoint} and {self.endPoint} """
        if self.startPoint < i < self.endPoint:
            return True

        if self.endPoint < self.startPoint:  # More than 360
            if i > self.endPoint or i < self.startPoint:
                return True

        return False

    def minimumSizeHint(self):
        return QtCore.QSize(15, 15)

    def sizeHint(self):
        return QtCore.QSize(336, 336)

    def mousePressEvent(self, event: QtGui.QMouseEvent):
        self.update_pos(event)
        super(PathSlider, self).mousePressEvent(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent):
        self.update_pos(event)
        super(PathSlider, self).mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent):
        self.dragging = None
        self.rangeChanged.emit(self.startPoint, self.endPoint)
        super().mouseReleaseEvent(event)
