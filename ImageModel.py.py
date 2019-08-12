# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 17:45:16 2019

@author: Brian
"""

from PIL import Image
import math
import random
import numpy as np 

class SimpleNN:
    def __init__(self, depth, layer_size):
        IMAGE_SIZE = 784
        CLASSIFICATION_RANGE = 10
        
        self.neuralNet = {'input': [None for i in range(IMAGE_SIZE)], 
                          'layers':[[None for i in range(layer_size)] for j in range(depth)],
                          'output': [None for i in range(CLASSIFICATION_RANGE)]}
        self.wmatrix1 = [[random.random() for i in range(IMAGE_SIZE)] for j in range(layer_size)]
        
    def loadImage(self, image):
        pixels = Image.open(image).getdata()
        for pixel in pixels:
            self.neuralNet['input'].append(pixel)
        
        
        
    # For the backpropagation stage   
    
    def computeGradient(self):
        pass
        
    def doGradientDescent(self, vector, gradient):
        for i in range(len(vector)):
            vector[i] -= gradient[i]
            
    def updateVariables(self):
        pass


