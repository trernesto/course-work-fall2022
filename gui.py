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
      self.unitUI()
      self.resize(width, height)
      self.nn = neural_network()
      self.data = self.nn.data
      
   #painter creating
   def paintEvent(self, event):
      painter = QPainter(self)
      #nn drawing
      font = QFont()
      font.setFamily("Arial")
      font.setPointSize(16)
      if (self.modified):
         self.nn_draw(painter, 100, -125)

   def unitUI(self):
      button = QPushButton("start nn", self)
      button.move(100, 0)
      button.clicked.connect(self.buttonClicked1)

      button = QPushButton("results", self)
      button.move(100, 100)
      button.clicked.connect(self.buttonClicked2)

   def buttonClicked1(self):
      self.modified = True
      t = Thread(target=self.doCalculations)
      t.start()

   def doCalculations(self):
      for i in range(5):
         self.nn._250_epoch()
         self.data = self.nn.bias
         self.update()
         time.sleep(1)

   def buttonClicked2(self):
      self.nn.print_result()

   def nn_draw(self, painter, start_width = 0, start_height = 0):
      data = self.data
      sw = start_width
      sh = start_height
      for column in data:
         sh = start_height
         sw += 200
         for element in column:
            sh += 150
            self.node_draw(painter, element, sw, sh)

   def node_draw(self, painter, element, start_width, start_height):
      painter.setPen(QPen(Qt.blue,  4, Qt.SolidLine))
      painter.drawEllipse(start_width, start_height, 100, 100)
      painter.setPen(QPen(Qt.green))
      #Draw numbers in nodes 
      painter.setFont(QFont("Arial", 30))
      painter.drawText(start_width + 13, start_height + 65, str(element)[1:5])

   def neuron_draw(self, painter, start_width, start_height):
      print(1)


def create_window():
   app = QApplication(sys.argv)
   ex = window()
   ex.show()
   sys.exit(app.exec_())

