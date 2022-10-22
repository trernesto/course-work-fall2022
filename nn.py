import numpy as np
class neural_network():
	input_layer = np.array([1, 1, 0, 0, 0])

	def __init__ (self):
		number_of_hidden_layers = 2
		size_of_hidden_layer = 3
		sizes = generate_sizes_of_nn(number_of_hidden_layers, size_of_hidden_layer)
		theta = []
		bias = generate_rand_bias(sizes)
		data  = np.array([0, 1], dtype = object)
		theta.append(generate_rand_theta(2, 3))
		theta.append(generate_rand_theta(3, 3))
		theta.append(generate_rand_theta(3, 1))
		#feed_forward(data, theta, bias)
		backpropagation(data, 0, theta, bias, 0.3, sizes)

#matrix multiplying in np python realized by @ operand
#z = theta @ X.transpose()		
def sigmoid(z):
	if (type(z) is not float):
		z = z.astype(float)
	sig = 1/(1 + np.exp(-z))
	return sig

def generate_sizes_of_nn(number_of_hidden_layers, 
	size_of_hidden_layer, size_of_output_layer = 1):
	sizes = [size_of_hidden_layer] * number_of_hidden_layers
	sizes.append(size_of_output_layer)
	return sizes

#generate random biases from 0 to 1
def generate_rand_bias(sizes):
	#https://numpy.org/doc/stable/reference/random/generated/numpy.random.randn.html?highlight=randn#numpy.random.randn
	bias = [np.random.randn(i, 1) for i in sizes[:]]
	return bias

#generate random weights from 0 to 1
#width is a number of input neurons, 
#height is a number of output neurons  
def generate_rand_theta(width, height):
	#123 in this case is a seed of rng
	array = []
	rng = np.random.default_rng(123)
	for i in range(width * height):
		array.append(rng.random())
	return np.array(array).reshape(height, width)

def cost_function(our_value, target_value):
	cost = np.square(our_value - target_value)
	return np.sum(cost)/len(cost)

def feed_forward(data, theta, bias):
	a = []
	z = []
	z.append(theta[0] @ data.transpose() + bias[0].flatten())
	a.append(sigmoid(z[0]))
	for i in range(1, len(theta)):
		z.append(theta[i] @ a[i - 1] + bias[i].flatten())
		a.append(sigmoid(z[i]))
	print(a[-1])

def backpropagation(data, y, theta, bias, alpha, sizes):
	#feed forward to compute activasions
	a = []
	z = []
	z.append(theta[0] @ data.transpose() + bias[0].flatten())
	a.append(sigmoid(z[0]))
	for i in range(1, len(theta)):
		z.append(theta[i] @ a[i - 1] + bias[i].flatten())
		a.append(sigmoid(z[i]))
	print(a[-1])

	#count the first derivative and going backwards
	#dC/dTheta(i) = dC/da(i) * da(i)/dz(i) * dz(i)/dw(i)
	#dC/da = ((a(i) - y)^2)' = 2(a(i) - y)
	#da/dz = (sigmoid(z))' = sigmoid(z) * (1 - sigmoid(z))
	#dz/dw = (w(i) * a(i))' = a(i)
	#delta = dC/d(bias) = dC/da * da/dz * dz/d(bias)
	#dz/d(bias) = 1
	delta = (a[-1] - y) * (sigmoid(z[-1]) * (1 - sigmoid(z[-1])))
	grad_bias = [np.zeros(b.shape) for b in bias]
	grad_theta = [np.zeros(t.shape) for t in theta]
	grad_bias[-1] = delta
	grad_theta[-1] = delta @ a[-2].reshape(1, a[-2].size)
	for i in range(2, len(sizes)):
		dz = sigmoid(z[-i]) * (1 - sigmoid(z[-i]))
		delta = (theta[-i + 1].transpose() @ delta) * dz
		grad_bias[-i] = delta
		grad_theta[-i] = delta.reshape(delta.size, 1) @ a[-i - 1].reshape(1, a[-i - 1].size)

def train(data, y, theta):
	print("123")