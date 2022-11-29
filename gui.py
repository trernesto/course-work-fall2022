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
      self.data_number = 0
      self.data_number_test = 0
      super(window, self).__init__(parent)
      self.setStyleSheet("background-color : darkgray;")
      self.unitUI()
      self.resize(width, height)
      self.nn = neural_network()
      self.data = self.nn.data[self.data_number]
      self.a = self.nn.getActiv(self.data)
      self.text = ""
      self.clicked_x = ""
      self.clicked_y = ""
      
   #painter creating
   def paintEvent(self, event):
      painter = QPainter(self)
      #nn drawing
      font = QFont()
      font.setFamily("Arial")
      font.setPointSize(16)
      self.nn_draw(painter, 100, -125)

      painter.setFont(QFont("Arial", 35))
      painter.setPen(Qt.black);
      sw = 50
      sh = 800
      painter.drawText(sw, sh, "Accuracy: " + self.text)

      painter.setFont(QFont("Arial", 24))
      painter.setPen(QPen(Qt.white,  4, Qt.SolidLine))
      painter.drawText(50, 950, self.clicked_x)
      painter.drawText(50, 1000, self.clicked_y)

   def unitUI(self):
      button = QPushButton("start nn", self)
      button.move(50, 0)
      button.resize(150,75)
      button.setStyleSheet("background-color: white")
      button.clicked.connect(self.buttonClicked1)

      button = QPushButton("results", self)
      button.move(50, 100)
      button.resize(150,75)
      button.setStyleSheet("background-color: white")
      button.clicked.connect(self.buttonClicked2)

      button = QPushButton("restart", self)
      button.move(50, 200)
      button.resize(150,75)
      button.setStyleSheet("background-color: white")
      button.clicked.connect(self.buttonClicked3)

      button = QPushButton("next", self)
      button.move(50, 300)
      button.resize(150,75)
      button.setStyleSheet("background-color: white")
      button.clicked.connect(self.buttonClicked4)

      button = QPushButton("testSubj", self)
      button.move(50, 400)
      button.resize(150,75)
      button.setStyleSheet("background-color: white")
      button.clicked.connect(self.buttonClicked5)

      button = QPushButton("print Accuracy on test", self)
      button.move(50, 500)
      button.resize(150,75)
      button.setStyleSheet("background-color: white")
      button.clicked.connect(self.buttonClicked6)

   def buttonClicked1(self):
      t = Thread(target=self.doCalculations)
      t.start()

   def doCalculations(self):
      self.text = ""
      for i in range(4):
         self.nn._250_epoch()
         self.data = self.nn.data[self.data_number]
         self.a = self.nn.getActiv(self.data)
         self.update()
         time.sleep(1)

   def buttonClicked2(self):
      self.text = str(round(self.nn.accuracy, 5))
      self.update()

   def buttonClicked3(self):
      self.nn = neural_network()
      self.data = self.nn.data[self.data_number]
      self.a = self.nn.getActiv(self.data)
      self.update()

   def buttonClicked4(self):
      number_of_elements_in_data = self.nn.data.shape[0]
      self.data_number = (self.data_number + 1) % number_of_elements_in_data
      self.data = self.nn.data[self.data_number]
      self.a = self.nn.getActiv(self.data)
      self.update()

   def buttonClicked5(self):
      number_of_elements_in_data = self.nn.testdata.shape[0]
      self.data_number_test = (self.data_number_test + 1) % number_of_elements_in_data
      self.data = self.nn.testdata[self.data_number_test]
      self.a = self.nn.getActiv(self.data)
      print(self.a)
      self.update()

   def buttonClicked6(self):
      loss = []
      y = self.nn.testy
      data = self.nn.testdata
      theta = self.nn.theta
      bias = self.nn.bias
      i = 0
      for element in data:
         nn_value = self.nn.getActiv(element)[-1]
         loss.append(self.nn.cf(nn_value, y[i]))
         i += 1
      accuracy = 1 - sum(loss)/len(data)
      print(accuracy)


   def nn_draw(self, painter, start_width = 0, start_height = 0):
      sw = start_width
      sh = start_height
      data = self.data
      a = self.a
      self.neuron_draw(painter)
      sw += 200
      for element in data:
         sh += 150
         self.node_draw(painter, element, sw, sh)

      for column in a:
         sh = start_height
         sw += 200
         for element in column:
            sh += 150
            self.node_draw(painter, element, sw, sh)

   def node_draw(self, painter, element, start_width, start_height):
      painter.setPen(QPen(Qt.black,  4, Qt.SolidLine))
      color = QColor(255, 255, 255, (int)((element)*255))
      painter.setBrush(QBrush(color, Qt.SolidPattern))
      painter.drawEllipse(start_width, start_height, 100, 100)
      #Draw numbers in nodes 
      painter.setFont(QFont("Arial", 30))
      painter.drawText(start_width + 13, start_height + 65, str(round(element,2)))

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

   def getPos(self, x, y):
      pos_x = x - 300
      pos_y = y - 25
      if (pos_x % 200 <= 100 and pos_y  % 150 <= 100):
         return(int(pos_x / 200), int(pos_y / 150))
      return(-1, -1)

   def mousePressEvent(self, event):
      x = event.x()
      y = event.y()
      #number of activation element 
      x, y = self.getPos(x, y)
      self.clicked_x = "left weights: "
      self.clicked_y = "right weights: "
      try:
         if (x > 0):
            for i in range(len(self.nn.theta[x - 1][y])):
               self.clicked_x += str(round(self.nn.theta[x - 1][y][i], 3))
               self.clicked_x += " "
         if (x < self.nn.size_of_network_w - 1):
            for element in self.nn.theta[x]:
               self.clicked_y += str(round(element[y],3))
               self.clicked_y += " "
      except:
         print("ERROR")
         print("x: ", x)
         print("y: ", y)
      self.update()

def create_window():
   app = QApplication(sys.argv)
   ex = window()
   ex.show()
   sys.exit(app.exec_())

