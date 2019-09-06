# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 17:45:16 2019

@author: Brian
"""

from PIL import Image
import time
import random
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def LRAdjust(x):
    return abs(np.exp(-x)*(np.cos(5*x) + np.sin(5*x)))

class SimpleNN:
    def __init__(self, depth, layer_size, class_range, image_size):
        # Miscellaneous variables
        self.class_range = class_range
        self.image_size = image_size
        self.memo = {} # For backpropagation DP
        self.expected = 3

        # Initialize all neurons
        self.neurons = [[random.random() for i in range(self.image_size)]]
        for i in range(depth):
            self.neurons.append([0 for i in range(layer_size)])
        self.neurons.append([0 for i in range(self.class_range)])
        self.size = len(self.neurons)

        # Initialize weights and biases
        self.biases = [[random.uniform(-3,3) for j in range(len(self.neurons[i+1]))] for i in range(self.size-1)]
        self.weights = [[[random.uniform(-3,3) for k in range(len(self.neurons[i]))]
                                          for j in range(len(self.neurons[i+1]))]
                                          for i in range(self.size - 1)]

    # The feed-forward stage
    def loadImage(self, image):
        pixels = Image.open(image).getdata()
        for i in range(self.image_size):
            self.neurons[0][i] = pixels[i]

    def feedForward(self):
        for i in range(self.size - 1):
            a = np.matmul(self.weights[i], self.neurons[i]) + self.biases[i]
            self.neurons[i+1] = sigmoid(a)

    def calculateError(self):
        error = 0
        for i in range(self.class_range):
            diff = 1 if i == self.expected else 0
            error += (diff-self.neurons[-1][i])**2
        return error

    # Helper functions to calculate partial derivatives for backprop
    def dcda(self, a_index):
        target_neuron = self.neurons[-1][a_index]
        diff = 1 if a_index == self.expected else 0
        return 2*(target_neuron - diff)

    def dada(self, layer, index1, index2):
        weight = self.weights[layer-1][index1][index2]
        activation = self.neurons[layer][index1]
        return weight*activation*(1-activation)

    def dadw(self, layer, source_index, target_index):
        src_neuron = self.neurons[layer-1][source_index]
        trgt_neuron = self.neurons[layer][target_index]
        return src_neuron*trgt_neuron*(1-trgt_neuron)

    def dadb(self, layer, neuron_index):
        neuron = self.neurons[layer][neuron_index]
        return neuron*(1-neuron)

     
    def DPPartial(self, *args):
        """
        Checks if a particular derivative is memoized. If not, compute the derivative.
        Parameters:
            deriv_index: a tuple containing info about a particular partial derivative
        Returns the value of that partial derivative
        """
        if args in self.memo:
            result = self.memo[args]
        else:
            if args[0] == 'C':
                result = self.dcda(args[1])
            else:
                result = self.dada(args[1], args[2], args[3])
            self.memo[args] = result
        return result

    def backProp(self, variable, n = 0, a_index = 0):
        """
        Possibly the most important helper of all...

        Computes and returns the partial derivative of C, the cost function with respect to
        a given variable.
        
        Parameters:
            variable: a tuple containing a string (w or b) and another tuple with length of either 2 or 3
            depending on the variable type whose elements denote the indices of our variable.
        """
        var_type, var_coordinate = variable
        layer_level = self.size-n
        layer_size = len(self.neurons[layer_level-1])
        neuron_index = var_coordinate[1] 
        result = 0

        # Special trivial case where weight or bias layer is the last one
        if self.size - var_coordinate[0] == 2:
            partial = self.DPPartial('C',neuron_index)
            if var_type == 'w':
                return partial * self.dadw(-1, var_coordinate[2], neuron_index)
            else:
                return partial * self.dadb(-1, neuron_index)

        # Base case
        if layer_level - var_coordinate[0] == 2:
            partial = self.DPPartial('a',layer_level,a_index,neuron_index) 
            if var_type == 'w':
                return partial*self.dadw(layer_level-1, var_coordinate[2], neuron_index)
            else:
                return partial*self.dadb(layer_level-1, neuron_index)

        # Recursive case
        else:
            for i in range(layer_size):
                # Distinguishes between starting and general recursive case
                derivative_index = ('C',i) if n == 0 else ('a',layer_level,a_index,i) 
                partial = self.DPPartial(*derivative_index)
                result += partial * self.backProp(variable, n+1, i)
            return result
        
    def doGradientDescent(self, batch_size):
        for i in range(self.size - 1):
            matrix = self.weights[i]
            bias_array = self.biases[i]
            # Update biases according to gradient
            for j in range(len(bias_array)):
                yield ('b', i, j, self.backProp(('b', (i,j)))/batch_size)
            for j in range(len(matrix)):
                row = matrix[j]
                saved_grad_term = self.backProp(('w', (i,j,0)))/batch_size
                first_neuron = self.neurons[i][0]
                # Update weights according to gradient
                for k in range(len(row)):
                    if k == 0:
                        result = saved_grad_term
                    else:
                        result = (self.neurons[i][k]/first_neuron) * saved_grad_term
                    yield ('w', i, j, k, result)

#start = time.time()
#print('hello world')
#end = time.time()
#print(end - start)

"""
TESTING BACKPROPAGATION
"""

#a = SimpleNN(2, 2, 3, 2)
#a.feedForward()
#def calculate_test1():
#    t = 0
#    dadw = a.neurons[0][0]*a.neurons[1][1]*(1-a.neurons[1][1])
#    for i in range(3):
#        subtotal = 0
#        dcda = a.dcda(i)
#        for j in range(2):
#            a3a2 = a.weights[2][i][j]*a.neurons[3][i]*(1-a.neurons[3][i]) 
#            a2a1 = a.weights[1][j][1]*a.neurons[2][j]*(1-a.neurons[2][j]) 
#            subtotal += dadw*a3a2*a2a1
#        t += dcda*subtotal
#    return t
#
#def calculate_test2():
#    t = 0
#    dadw = a.neurons[1][1]*a.neurons[2][1]*(1-a.neurons[2][1])
#    for i in range(3):
#        dcda = 2*(a.neurons[3][i]-1) if i == a.expected else 2*a.neurons[3][i]
#        dada = a.neurons[3][i]*(1-a.neurons[3][i])*a.weights[2][i][1]
#        t += dcda*dada*dadw])
#   
#print(abs(calculate_test1() - a.backProp(('w', (0,1,0)))) < .00000000001)
#print(abs(calculate_test2() - a.backProp(('w', (1,1,1)))) < .00000000001)
#print(abs(calculate_test3() - a.backProp(('w', (2,2,0)))) < .00000000001)

#    return t
#
#
#def calculate_test3():
#    return 2*(a.neurons[3][2]-1)*a.neurons[2][0]*a.neurons[3][2]*(1-a.neurons[3][2])
"""
TESTING LEARNING ON 1 SAMPLE
"""
costs = []
start = time.time()
b = SimpleNN(2,16,10,784)
for i in range(500):
    b.feedForward()
    gradients = b.doGradientDescent(1)
    if i % 50 == 0:
        print('Cost at round {}: '.format(i), b.calculateError())
    for j in gradients:
        if j[0] == 'w':
            b.weights[j[1]][j[2]][j[3]] -= j[-1]
        else:
            b.biases[j[1]][j[2]] -= j[-1]
    costs.append(b.calculateError())
            
print('------------------')
print('Result: ')
for l in range(10):
    if l == 0:
        print('[' + str(b.neurons[-1][l]) + ',')    
    elif l == 9:
        print(str(b.neurons[-1][l]) + ']')
    else:
        print(str(b.neurons[-1][l]) + ',')
print('Final cost: ', b.calculateError())
classified = np.argmax(b.neurons[-1])
print('Classified as: ', classified, ', Activation value: ', b.neurons[-1][classified])
        
end = time.time()
print('Time to complete: ' + str(end-start) + ' seconds')

f1 = plt.figure()
plt.plot([i for i in range(500)], costs)
plt.title('Cost over training steps')
plt.xlabel("Training steps")
plt.ylabel("Cost")

f2 = plt.figure()
plt.bar([i for i in range(10)], b.neurons[-1])
plt.title('Final layer activatations')
plt.xlabel('Integer range')
plt.ylabel('Activation degree')

plt.show()

