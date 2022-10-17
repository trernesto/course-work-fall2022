import numpy as np
class neural_network():
	input_layer = np.array([1, 1, 0, 0, 0])
	number_of_hidden_layers = 2
	size_of_hidden_layer = 3
	output_layer = np.array([0, 0])

	def __init__ (self):
		theta = []
		data  = np.array([0, 1], dtype = object)
		theta.append(generate_rand_theta(3, 2))
		theta.append(generate_rand_theta(1, 3))
		feed_forward(data, theta)

#matrix multiplying in np python realized by @ operand
#z = theta @ X.transpose()		
def sigmoid(z):
	if (type(z) is not float):
		z = z.astype(float)
	sig = 1/(1 + np.exp(-z))
	return sig

def feed_forward(data, theta):
	a = []
	temp = sigmoid(theta[0] @ data.transpose())
	a.append(temp.transpose())
	for i in range(1, len(theta)):
		temp = sigmoid(theta[i] @ a[i - 1])
		a.append(temp.transpose())
	print(a[len(theta) - 1])

#generate random weights from 0 to 1  
def generate_rand_theta(height, width):
	#123 in this case is a seed of rng
	array = []
	rng = np.random.default_rng(123)
	for i in range(width * height):
		array.append(rng.random())
	return np.array(array).reshape(height, width)


def backpropagation():
	print("TO DO")