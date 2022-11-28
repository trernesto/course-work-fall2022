import sys
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import *
import numpy as np
from nn import neural_network
import time
from threading import *

class window(QWidget):
   def __init__(self, parent = None, width = 1920, height = 1080):
      self.modified = False
      super(window, self).__init__(parent)
      self.setStyleSheet("background-color : black;")
      self.unitUI()
      self.resize(width, height)
      self.nn = neural_network()
      self.data = self.nn.data
      self.a = self.nn.a
      self.text = ""
      
   #painter creating
   def paintEvent(self, event):
      painter = QPainter(self)
      #nn drawing
      font = QFont()
      font.setFamily("Arial")
      font.setPointSize(16)
      self.nn_draw(painter, 100, -125)

      painter.setFont(QFont("Arial", 13))
      painter.setPen(Qt.white);
      sw = 1250
      sh = 500
      for i in range((int)(len(self.text) / 3)):
         painter.drawText(sw, sh, str(self.text[3*i:3*i+3]))
         sh += 25

   def unitUI(self):
      button = QPushButton("start nn", self)
      button.move(50, 0)
      button.resize(150,100)
      button.setStyleSheet("background-color: white")
      button.clicked.connect(self.buttonClicked1)

      button = QPushButton("results", self)
      button.move(50, 100)
      button.resize(150,100)
      button.setStyleSheet("background-color: white")
      button.clicked.connect(self.buttonClicked2)

      button = QPushButton("restart", self)
      button.move(50, 200)
      button.resize(150,100)
      button.setStyleSheet("background-color: white")
      button.clicked.connect(self.buttonClicked3)

   def buttonClicked1(self):
      t = Thread(target=self.doCalculations)
      t.start()

   def doCalculations(self):
      for i in range(4):
         self.nn._250_epoch()
         self.data = self.nn.data
         self.a = self.nn.a
         self.update()
         time.sleep(1)

   def buttonClicked2(self):
      self.text = self.nn.print_result()
      for i in range((int)(len(self.text) / 3)):
         print(self.text[3*i:3*i+3])
      self.update()

   def buttonClicked3(self):
      self.nn = neural_network()
      self.data = self.nn.data
      self.a = self.nn.a
      self.update()

   def nn_draw(self, painter, start_width = 0, start_height = 0):
      sw = start_width
      sh = start_height
      data = self.data
      a = self.a
      sw += 200
      for element in data[0]:
         sh += 150
         self.node_draw(painter, element, sw, sh)

      for column in a:
         sh = start_height
         sw += 200
         for element in column:
            sh += 150
            self.node_draw(painter, element, sw, sh)
      self.neuron_draw(painter)

   def node_draw(self, painter, element, start_width, start_height):
      painter.setPen(QPen(Qt.black,  4, Qt.SolidLine))
      color = QColor(255, 255, 255, (int)((1 - element)*255))
      painter.setBrush(QBrush(color, Qt.SolidPattern))
      painter.drawEllipse(start_width, start_height, 100, 100)
      #Draw numbers in nodes 
      painter.setFont(QFont("Arial", 30))
      painter.drawText(start_width + 13, start_height + 65, str(1 - element)[0:4])

   def neuron_draw(self, painter):
      sh = 25
      sw = 300 - 200
      sh_e = 25 - 150
      sw_e = 500 - 200
      for column in self.nn.theta:
         sw += 200
         sw_e += 200
         sh_e = 25 - 150
         for row in column:
            sh = 25
            sh_e += 150
            for element in row:
               if (element < 0):
                  painter.setPen(QPen(Qt.red, abs(element), Qt.SolidLine))
               else:
                  painter.setPen(QPen(Qt.green, element, Qt.SolidLine))
               painter.drawLine(sw + 100, sh + 50, sw_e, sh_e + 50)
               sh += 150

def create_window():
   app = QApplication(sys.argv)
   ex = window()
   ex.show()
   sys.exit(app.exec_())

