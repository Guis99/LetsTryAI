# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 17:45:16 2019

@author: Brian
"""

from PIL import Image
import math
import random
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoidprime(x):
    return sigmoid(x) * (1 - sigmoid(x))

class SimpleNN:
    def __init__(self, depth, layer_size, class_range, image_size):
        # Miscellaneous variables
        self.class_range = class_range
        self.image_size = image_size
        self.memo = {} # For backpropagation DP
        self.random_storage = random.uniform(-5, 5)
        self.expected = 2

        # Initialize all neurons
        self.neurons = [[1 for i in range(self.image_size)]]
        for i in range(depth):
            self.neurons.append([0 for i in range(layer_size)])
        self.neurons.append([0 for i in range(self.class_range)])
        self.size = len(self.neurons)

        # Initialize weights and biases
        self.weights = [[[.2 for k in range(len(self.neurons[i]))]
                                          for j in range(len(self.neurons[i+1]))]
                                          for i in range(self.size - 1)]

        self.biases = [[1 for j in range(len(self.neurons[i+1]))] for i in range(self.size-1)]


    # The feed-forward stage
    def loadImage(self, image):
        pixels = Image.open(image).getdata()
        for i in range(self.image_size):
            self.neurons[0][i] = pixels[i]

    def feedForward(self):
        for i in range(self.size - 1):
            a = np.matmul(self.weights[i], self.neurons[i]) + self.biases[i]
            self.neurons[i+1] = sigmoid(a)

    def calculateError(self, expected):
        error = 0
        for i in range(self.class_range):
            if i == expected:
                error += (1-self.neurons[-1][i])**2
            else:
                error += self.neurons[-1][i]**2
        return error


    # For the backpropagation stage
    def backProp(self, variable, n = 0, result = 0, a_index = 0):
        """
        Possibly the most important helper of all...

        Parameters:
            variable: a tuple containing a string (w or b) and another tuple with length of either 2 or 3
            depending on the variable type whose elements denote the indices of our variable.

        Computes and returns the partial derivative of C, the cost function with respect to
        a given variable.
        """
        var_type = variable[0]
        var_coordinate = variable[1]
        layer_level = self.size-n
        layer_size = len(self.neurons[layer_level-1])
        # Special trivial case where weight or bias layer is the last one
        if self.size - var_coordinate[0] == 2:
            derivative_index = ('C','a',var_coordinate[1])
            neuron_index = var_coordinate[1]
            target_neuron = self.neurons[-1][neuron_index]
            if derivative_index in self.memo:
                partial = self.memo[derivative_index]
            else:
                partial = 2*(target_neuron-1) if neuron_index == self.expected else 2*target_neuron
                self.memo[derivative_index] = partial
            if var_type == 'w':
                return partial * target_neuron * self.neurons[-2][var_coordinate[2]] * (1 - target_neuron)
            else:
                return partial * target_neuron * (1 - target_neuron)

        # Base case
        if layer_level - var_coordinate[0] == 2:
            current_neuron = self.neurons[layer_level][a_index]
            if var_type == 'w':
                return current_neuron*self.neurons[layer_level-1][var_coordinate[2]]*(1-current_neuron)
            else:
                return current_neuron*(1-current_neuron)
        # Starting recursive case
        elif n == 0:
            for i in range(len(self.neurons[layer_level-1])):
                derivative_index = ('C','a',i)
                if derivative_index in self.memo:
                    partial = self.memo[derivative_index]
                else:
                    partial = 2*(1-self.neurons[layer_level-1][i]) if i == self.expected else 2*self.neurons[layer_level-1][i]
                    self.memo[derivative_index] = partial

                result += partial * self.backProp(variable, n+1, 0, i)
            return result
        # General recursive case
        else:
            for i in range(layer_size):
                derivative_index = ('a',layer_level, a_index, i)
                if derivative_index in self.memo:
                    partial = self.memo[derivative_index]
                else:
                    a_current = self.neurons[layer_level][a_index]
                    partial = a_current*self.weights[layer_level-1][a_index][i]*(1-a_current)
                    self.memo[derivative_index] = partial
                result += partial * self.backProp(variable, n+1, 0, i)
            return result

    def doGradientDescent(self, batch_size):
        for i in range(self.size - 1):
            matrix = self.weights[i]
            bias_array = self.biases[i]
            # Update biases according to gradient
            for j in range(len(bias_array)):
                bias_array[j] -= (self.backProp(('b', (i,j)))/batch_size)
            for j in range(len(matrix)):
                row = matrix[j]
                # Update weights according to gradient
                for k in range(len(row)):
                    row[k] -= (self.backProp(('w', (i,j,k)))/batch_size)


def train(epochs, batch_size):
    pass



a = SimpleNN(2, 2, 3, 2)
a.feedForward()
def calculate_test1():
    t = 0
    for i in range(3):
        dcda = 2*(a.neurons[3][i]-1) if i == 2 else 2*a.neurons[3][i]
        for j in range(2):
            a3j = a.neurons[3][j]
            a2j = a.neurons[2][j]
            partialaw = a.neurons[0][0]*a.neurons[1][1]*(1-a.neurons[1][1])
            partial = partialaw * a.weights[2][i][j]*a.weights[1][j][1]*a3j*(1-a3j)*a2j*(1-a2j)
            t += dcda * partial
    return t

def calculate_test2():
    return 2*(a.neurons[3][2]-1)*a.neurons[2][0]*a.neurons[3][2]*(1-a.neurons[3][2])

print(abs(calculate_test1() - a.backProp(('w', (0,1,0)))) < .00000000001)
a.memo = {}
print(a.backProp(('w', (2,2,0))), calculate_test2())
print(abs(calculate_test2() - a.backProp(('w', (2,2,0)))) < .00000000001)
