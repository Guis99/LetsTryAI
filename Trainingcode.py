# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 19:01:48 2019

@author: Brian
"""

import ImageModel as nn
import numpy as np
import matplotlib.pyplot as plt
import time


training_data = []

def do_training(rounds, batch_size, layers, layer_size, class_range, input_size):
    network = nn.SimpleNN(layers, layer_size, class_range, input_size)
    for i in range(rounds):
        run_training_one_round(batch_size, layers, layer_size, class_range, input_size, network)
        

def run_training_one_round(batch_size, layers, layer_size, class_range, input_size, network):
    biases = np.array([[0 for j in range(len(network.neurons[i+1]))] for i in range(network.size-1)])
    weights = np.array([[[0 for k in range(len(network.neurons[i]))]
                                          for j in range(len(network.neurons[i+1]))]
                                          for i in range(network.size - 1)])
    for i in range(batch_size):
        network.loadImage(training_data[i])
        network.feedForward()
        gradients = network.doGradientDescent(batch_size)
        for j in gradients:
            if j[0] == 'w':
                weights[j[1]][j[2]][j[3]] += j[-1]
            else:
                biases[j[1]][j[2]] += j[-1]
    network.biases -= biases
    network.weights -= weights
        
        
"""
TESTING LEARNING ON 1 SAMPLE
"""
nn = nn.SimpleNN(2, 16, 10, 784)
costs = []

start = time.time()
for i in range(1000):
#    LR = 1 if i > 10 else 12
    nn.feedForward()
    gradients = nn.doGradientDescent()
    if i % 50 == 0:
        print('Cost at round {}: '.format(i), nn.calculateError())
    for j in gradients:
        if j[0] == 'w':
            nn.weights[j[1]][j[2]][j[3]] -= j[-1]
        else:
            nn.biases[j[1]][j[2]] -= j[-1]
    costs.append(nn.calculateError())
    nn.memo = {}
end = time.time()          
print('------------------')
print('Results:')
classified = np.argmax(nn.neurons[-1])
print('Classified as: ', classified, ', Activation value: ', nn.neurons[-1][classified])
if classified == nn.expected:
    print('The network correctly identified the image!')
else:
    print('Incorrect. The network misclassified the image as a ' + str(classified) + ' when it was a ' + str(nn.expected))
print('Final cost: ' + str(nn.calculateError()))
end = time.time()
print('Time to complete: ' + str(end-start) + ' seconds')

f1 = plt.figure()
plt.plot([i for i in range(1000)], costs)
plt.title('Cost over training steps')
plt.xlabel("Training steps")
plt.ylabel("Cost")

f2 = plt.figure()
plt.bar([i for i in range(4)], nn.neurons[-1])
plt.title('Final layer activatations')
plt.xticks([i for i in range(4)])
plt.xlabel('Activation range')
plt.ylabel('Activation degree')

plt.show()