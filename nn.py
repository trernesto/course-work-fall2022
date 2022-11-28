import numpy as np
from data import component

class neural_network():
	input_layer = np.array([1, 1, 0, 0, 0])

	def __init__ (self):
		self.epochCounter = 0
		#COMPONENT CREATION BLOCK
		number_of_hidden_layers = 4
		size_of_hidden_layer = 6
		sizes = generate_sizes_of_nn(number_of_hidden_layers, size_of_hidden_layer)
		theta = []
		bias = generate_rand_bias(sizes)
		components = component()
		data, y = components.generateXOR()
		theta.append(generate_rand_theta(2, size_of_hidden_layer))
		for i in range(number_of_hidden_layers - 1):
			theta.append(generate_rand_theta(size_of_hidden_layer, size_of_hidden_layer))
		theta.append(generate_rand_theta(size_of_hidden_layer, 1))

		self.sizes = sizes
		self.data = data
		self.y = y
		self.bias = bias
		self.theta = theta
		self.a = feed_forward(self.data[i], self.theta, self.bias)

#TRAINING
	def print_result(self):
		a = []
		result = ''
		for i in range(len(self.data)):
			a.append(feed_forward(self.data[i], self.theta, self.bias))
			tup = ("our value: ", a[i][-1], " target value: ", self.y[i], 
				" difference: ", a[i][-1] - self.y[i])
			for item in tup:
				if (type(item) is not str):
					result = result + np.array_str(item)
				else:
					result = result + item
			result = result + '\n'
		return result

	def _250_epoch(self):
		sizes = self.sizes
		data = self.data
		y = self.y
		bias = self.bias
		theta = self.theta
		self.bias, self.theta = self.train(data, y, theta, 
			bias, sizes, 1.3, 250)

	def train(self, data, y, theta, bias, sizes, alpha = 0.03, epoch = 300):
		for epo in range(epoch + 1):
			loss = []
			for i in range(len(data)):
				nn_value = feed_forward(data[i], theta, bias)[-1]
				loss.append(cost_function(nn_value, y[i]))
				grad_bias, grad_theta = backpropagation(data[i], y[i], theta, bias, sizes)
				for j in range(len(bias)):
					bias[j] = bias[j] - alpha * grad_bias[j].reshape(grad_bias[j].size, 1)
				for j in range(len(theta)):
					theta[j] = theta[j] - alpha * grad_theta[j]
			accuracy = 1 - sum(loss)/len(data)
			if (self.epochCounter % 250 == 0):
				self.a = feed_forward(data[i], theta, bias)
				print("epoch: ", self.epochCounter, "    acc: ", accuracy)
			self.epochCounter += 1
		return (bias, theta)
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
	return np.random.randn(height, width)

def cost_function(our_value, target_value):
	cost = np.square(our_value - target_value)
	return np.sum(cost)/len(target_value)

def feed_forward(data, theta, bias):
	a = []
	z = []
	z.append(theta[0] @ data.transpose() + bias[0].flatten())
	a.append(sigmoid(z[0]))
	for i in range(1, len(theta)):
		z.append(theta[i] @ a[i - 1] + bias[i].flatten())
		a.append(sigmoid(z[i]))
	return a

def backpropagation(data, y, theta, bias, sizes):
	#feed forward to compute activations
	a = [data]
	z = []
	z.append(theta[0] @ data.transpose() + bias[0].flatten())
	a.append(sigmoid(z[0]))
	for i in range(1, len(theta)):
		z.append(theta[i] @ a[i] + bias[i].flatten())
		a.append(sigmoid(z[i]))

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
	for i in range(2, len(sizes) + 1):
		dz = sigmoid(z[-i]) * (1 - sigmoid(z[-i]))
		delta = (theta[-i + 1].transpose() @ delta) * dz
		grad_bias[-i] = delta
		grad_theta[-i] = delta.reshape(delta.size, 1) @ a[-i - 1].reshape(1, a[-i - 1].size)
	return(grad_bias, grad_theta)
