# -*- coding: utf-8-*-
"""
Module : simple_QPanda3D_example
Author : Saifeddine ALOUI
Description :
    This is an example of how we can put togather a simple Panda3D Word
    wrapped inside a QMainWindow.
"""

from QPanda3D.Panda3DWorld import Panda3DWorld
from QPanda3D.QPanda3DWidget import QPanda3DWidget
# import PyQt5 stuff

from PyQt5.QtCore import *
from PyQt5.QtGui import *
import threading
from PyQt5.QtWidgets import *
import sys
from Pandas3D import textures, utils
from panda3d.core import ColorBlendAttrib, TransformState, ClockObject, PStatClient, WindowProperties
import numpy as np
from direct.showbase import ShowBaseGlobal
from direct.gui.OnscreenText import OnscreenText
from direct.task import Task
from panda3d.core import Point3, Vec3, Vec4, VBase4, Texture, TextureStage, CardMaker
import os


class PandaTest(Panda3DWorld):
    """
    This is the class that defines our world
    It inherits from Panda3DWorld that inherits from :1
    Panda3D's ShowBase class
    """

    def __init__(self):

        Panda3DWorld.__init__(self)

        # Create scenegraph, attach stimulus to card.
        cm = CardMaker('card')
        cm.setFrameFullscreenQuad()
        self.card = self.render.attachNewNode(cm.generate())
        self.cam.setPos(0, -4, 0)

        self.texture_stage = TextureStage("texture_stage")
        # Scale is so it can handle arbitrary rotations and shifts in binocular case
        ShowBaseGlobal.globalClock.setMode(ClockObject.MLimited)
        ShowBaseGlobal.globalClock.setFrameRate(30)

        self.card.setScale(np.sqrt(8))
        self.card.setColor((1, 1, 1, 1))  # makes it bright when bright (default combination with card is add)
        print(os.path.dirname(os.path.abspath(__file__)))
        f = open(os.path.dirname(os.path.abspath(__file__)) + '/stimmed.txt')
        self.list_of_lists = []
        self.t = 0
        for line in f:
            stripped_line = line.strip()
            line_list = stripped_line.split()
            self.list_of_lists.append(line_list)

        f.close()

        self.createCard(float(self.list_of_lists[self.t][2]), float(self.list_of_lists[self.t][3]),
                        float(self.list_of_lists[self.t][4]), float(self.list_of_lists[self.t][5]))

    def createCard(self, angle, velocity, spatial_frequency, contrast):
        sin_red_tex = textures.SinRgbTex(texture_size=512,
                                         spatial_frequency=spatial_frequency,
                                         rgb=(255, 0, 0),
                                         intensity=contrast)
        tex = sin_red_tex.get_texture()

        self.angle = angle
        self.velocity = velocity
        self.tex = tex

        self.card.setTexRotate(self.texture_stage, self.angle)
        self.card.setTexture(self.texture_stage, self.tex)

        if self.velocity != 0:
            # Add task to taskmgr to translate texture
            self.taskMgr.add(self.moveTextureTask, "moveTextureTask" + str(self.t))
            self.taskMgr.doMethodLater(2, self.taskStop, "taskStop")

    # Task for moving the texture
    def moveTextureTask(self, task):
        new_position = -task.time * self.velocity
        self.card.setTexPos(self.texture_stage, new_position, 0, 0)  # u, v, w
        return Task.cont

    def taskStop(self, taskStop):
        self.taskMgr.remove("moveTextureTask" + str(self.t))
        if (self.t != len(self.list_of_lists)):
            self.t = self.t + 1
            self.createCard(float(self.list_of_lists[self.t][2]), float(self.list_of_lists[self.t][3]),
                            float(self.list_of_lists[self.t][4]), float(self.list_of_lists[self.t][5]))


class Color(QWidget):

    def __init__(self, color):
        super(Color, self).__init__()
        self.setAutoFillBackground(True)

        palette = self.palette()
        palette.setColor(QPalette.Window, QColor(color))
        self.setPalette(palette)
