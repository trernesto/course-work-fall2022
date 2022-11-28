import sys
from PyQt5.QtCore import *
from PyQt5.QtGui import *
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
      
   #painter creating
   def paintEvent(self, event):
      painter = QPainter(self)
      #nn drawing
      font = QFont()
      font.setFamily("Arial")
      font.setPointSize(16)
      self.nn_draw(painter, 100, -125)

   def unitUI(self):
      self.label = QLabel()
      self.label.setText("1234")
      self.label.move(100, 500)
      self.label.setStyleSheet("color: white")

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
      for i in range(5):
         self.nn._250_epoch()
         self.data = self.nn.data
         self.a = self.nn.a
         self.update()
         time.sleep(1)

   def buttonClicked2(self):
      self.label.setText(self.nn.print_result())
      print(self.nn.print_result())

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
      for array in data:
         sh += 150
         self.node_draw(painter, 0., sw, sh)

      for column in a:
         sh = start_height
         sw += 200
         for element in column:
            sh += 150
            self.node_draw(painter, element, sw, sh)

   def node_draw(self, painter, element, start_width, start_height):
      painter.setPen(QPen(Qt.black,  4, Qt.SolidLine))
      color = QColor(255, 255, 255, (int)((1 - element)*255))
      painter.setBrush(QBrush(color, Qt.SolidPattern))
      painter.drawEllipse(start_width, start_height, 100, 100)
      #Draw numbers in nodes 
      painter.setFont(QFont("Arial", 30))
      painter.drawText(start_width + 13, start_height + 65, str(element)[0:4])

   def neuron_draw(self, painter, start_width, start_height):
      print(1)


def create_window():
   app = QApplication(sys.argv)
   ex = window()
   ex.show()
   sys.exit(app.exec_())

