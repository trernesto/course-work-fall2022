import numpy as np

class component():
	def __init__(self):
		print("")

	def generateXOR(self):
		#alpha = 1.3 epoch = 1000
		data  = np.array([
				[0, 0],
				[0, 1],
				[1, 0],
				[1, 1]], dtype = object)
		y = np.array([[0], [1], [1], [0]], dtype = object)
		return (data, y)

	def generate_sin(self):
		data = np.linspace(0, 6.28, 10).reshape(10, 1)
		y = np.sin(data)
		return (data, y)		

	def generate_round(self):
		data = np.array([])
		y = np.array([])
		#generate 2 coordinates and y = 0 or 1
		for i in range(50):
			print(1)
		return (data, y)