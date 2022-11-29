import numpy as np

class component():
	def __init__(self):
		print("")
	#IN THIS VERSION y CAN BE ONLY 0 OR 1 NOT [0 ... 0, 1, 0 ... 0]
	def generateXOR(self):
		#alpha = 1.3 epoch = 1000
		data  = np.array([
				[0, 0],
				[0, 1],
				[1, 0],
				[1, 1]], dtype = object)
		y = np.array([[0], [1], [1], [0]], dtype = object)
		return (data, y)

	def generateNotXOR(self):
		data  = np.array([
				[0, 0],
				[0, 1],
				[1, 0],
				[1, 1]], dtype = object)
		y = np.array([[1], [0], [0], [1]], dtype = object)
		return (data, y)		

	def generate_round(self):
		data = np.empty((50, 2))
		y = np.empty((50, 1))
		#generate 2 coordinates and y = 0 or 1
		for i in range(50):
			data[i] = 12 * np.random.rand(2) - 6
		for i in range(50):
			if (data[i][0] * data[i][0] + data[i][1] * data[i][1] <= 9):
				y[i] = 1
			else:
				y[i] = 0
		return (data, y)