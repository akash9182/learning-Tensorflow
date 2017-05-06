from __future__ import print_function
import numpy as np

number_of_inputs = 5 
x_data = np.array([1, 2, 3, 4, 5])
y_data = np.array([2, 4, 6, 8, 10])
weight = 1
bias = 1.0
learning_rate = 0.1

# The basic formula is Y = ( w * X ) + B
# Y is the outputs
# w is the weight
# X is the inputs
# B is the bias

for i in range(20):
	print ('Step', i , ':')
	output = (x_data  * weight) + bias

	difference = y_data - output

	gradient = 0
	for i in range(number_of_inputs):
		gradient += difference[i] * x_data[i]

	gradient /= number_of_inputs

	loss = (difference * difference) / number_of_inputs
	print ('Loss', np.sum(loss))

	weight = weight + (learning_rate * gradient)
	print ('Weight:' , weight, '\n')



