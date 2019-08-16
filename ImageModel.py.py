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
    return 1 / (1 + math.exp(-x))

def sigmoidprime(x):
    return sigmoid(x)*(1-sigmoid(x))

class SimpleNN:
    def __init__(self, depth, layer_size):
        IMAGE_SIZE = 3
        CLASSIFICATION_RANGE = 10

        # Initialize all neurons
        self.neurons = [[None for i in range(IMAGE_SIZE)]]
        for i in range(depth):
            self.neurons.append([None for i in range(layer_size)])
        self.neurons.append([None for i in range(CLASSIFICATION_RANGE)])
        self.size = len(self.neurons)

        # Initialize weights and biases
        self.weights = [[[random.random() for k in range(len(self.neurons[i]))]
                                          for j in range(len(self.neurons[i+1]))]
                                          for i in range(self.size - 1)]

        self.biases = [[random.random() for j in range(len(self.neurons[i+1]))] for i in range(self.size-1)]


    # The feed-forward stage
    def loadImage(self, image):
        pixels = Image.open(image).getdata()
        for pixel in pixels:
            self.neurons[0].append(pixel)

    def feedForward(self):
        pass


    # For the backpropagation stage
    def partialC(self, variable):
        """
        Possibly the most important helper of all...

        Parameters:
            variable: a tuple containing a string (w or b) and another tuple with length of either 2 or 3
            depending on the variable type whose elements denote the indices of our variable.

        Computes and returns the partial derivative of C, the cost function with respect to
        a given variable.
        """
        if variable[0] == 'w':
            pass
        else:
            pass

    def doGradientDescent(self):
        for i in range(self.size - 1):
            matrix = self.weights[i]
            bias_array = self.biases[i]
            # Update biases according to gradient
            for j in range(len(bias_array)):
                bias_array[j] -= self.partialC('b', (i,j))
            for j in range(len(matrix)):
                row = matrix[j]
                # Update weights according to gradient
                for k in range(len(row)):
                    self.weights[i][j][k] -= self.partialC(('w', (i,j,k)))
