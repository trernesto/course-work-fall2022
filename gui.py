import sys
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import numpy as np

class window(QWidget):
   def __init__(self, parent = None, width = 1920, height = 1080):
      super(window, self).__init__(parent)
      self.resize(width, height)
   #painter creating
   def paintEvent(self, event):
      painter = QPainter(self)
      #nn drawing
      font = QFont()
      font.setFamily("Arial")
      font.setPointSize(16)

      data = np.array([[1. ,2. ,3.], [4., 6.]], dtype = object)
      nn_draw(painter, data, 200, 200)



def create_window():
   app = QApplication(sys.argv)
   ex = window()
   ex.show()
   sys.exit(app.exec_())

def nn_draw(painter, data, start_width = 0, start_height = 0):
   sw = start_width
   sh = start_height
   for column in data:
      sh = start_height
      sw += 200
      for element in column:
         sh += 150
         node_draw(painter, element, sw, sh)

def node_draw(painter, data, start_width, start_height):
   painter.setPen(QPen(Qt.blue,  4, Qt.SolidLine))
   painter.drawEllipse(start_width, start_height, 100, 100)
   painter.setPen(QPen(Qt.green))
   #Draw numbers in nodes 
   painter.setFont(QFont("Arial", 30))
   painter.drawText(start_width + 27, start_height + 65, str(data))

def neuron_draw(painter, data, start_width, start_height):
   print(1)
