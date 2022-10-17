import numpy as np
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
class neural_network():
	input_layer = np.array([1, 1, 0, 0, 0])
	number_of_hidden_layers = 2
	size_of_hidden_layer = 3
	output_layer = np.array([0, 0])

	def __init__ (self):
		#first coefficient - number of i
		#second - number of output nodes
		#third - number of nodes that we multiplying with
		#theta = np.zeros((2, 3, 2))
		theta = []
		data  = np.array([0, 1], dtype = object)
		theta.append(generate_rand_theta(2, 3))
		theta.append(generate_rand_theta(3, 1))
		# theta.append(np.array(
		# 			[[0.1, 0.1],
		# 			[0.2, 0.2],
		# 			[0.3, 0.3]]
		# 			, dtype = object))
		# theta.append(np.array(
		# 			[0, 0, 1]
		# 			, dtype = object))
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
def generate_rand_theta(width, height):
	#123 in this case is a seed of rng
	array = []
	rng = np.random.default_rng(123)
	for i in range(width * height):
		array.append(rng.random())
	return np.array(array).reshape(height, width)


def backpropagation():
	print("TO DO")