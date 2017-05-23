#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: archit
"""

import math
import random


class Connection:
    def __init__(self, n, w = None):
        self.neuron = n
        self.weight = w
        self.dell = 0
    
    def set_weight(self, w):
        self.weight = w

class Neuron:
    def __init__(self, layer=None, identity=None):
        self.input = 0
        self.output = 0
        self.delta = 0
        self.connections = []
        self.index = (layer, identity)
        

    def add_connection(self, conn):
        self.connections.append(conn)
    
    def set_connections(self, connections):
        self.connections = connections
        
#where number_of_neurons = [3, 2] 
#layer 1 with 3 Neurons
#layer 2 with 2 Neurons
       
class NeuralNetwork:
    def __init__(self, number_of_inputs, number_of_outputs, number_of_neurons):
        self.n_inputs = number_of_inputs
        self.n_outputs = number_of_outputs
        self.n_hidden_neurons = number_of_neurons
        self.i_neurons = []
        self.hidden_neurons = []
        self.o_neurons = []
        self.__construct()  
        self.__make_connections()
        self.Targetoutput = []
        self.learning = 0.3
        self.input_data = []
        self.output_data = []
        self.regularization = 0

    def __construct(self):
        self.i_neurons = [Neuron(0, i) for i in range(self.n_inputs + 1)]
        #self.i_neurons[0].output = 1
    
        layer = 1
        for n in self.n_hidden_neurons:
            neurons_in_layer = [Neuron(layer, i) for i in range(n + 1)]
            #neurons_in_layer[0].output = 1                    
            self.hidden_neurons.append(neurons_in_layer)
            layer = layer + 1
            
        self.o_neurons = [Neuron(layer, i+1) for i in range(self.n_outputs)]
    
    def __make_connections(self):
        all_neurons = []
        all_neurons.append(self.i_neurons)
        
        for neuron_layer in self.hidden_neurons:
            all_neurons.append(neuron_layer)
        
        all_neurons.append(self.o_neurons)
        
        total_layers = len(all_neurons)
        
        for i in range(total_layers-1):
            current_layer = all_neurons[i]
            next_layer = all_neurons[i+1]
            
            for neuron in current_layer:
                x = 0 if i == (total_layers-2) else 1
                for next_neuron in next_layer[x:]:
                    neuron.add_connection(Connection(next_neuron, random.uniform(-1,1)))
        
    
    def set_weight(self, layer_id, neuron_id, next_neuron_id, w):
        if layer_id == 0:
            self.i_neurons[neuron_id].connections[next_neuron_id-1].set_weight(w)
        else:
            self.hidden_neurons[layer_id-1][neuron_id].connections[next_neuron_id-1].set_weight(w)
                    
     
    def __set_input(self, inputs):
        self.i_neurons[0].output = 1
        i = 1
        for input in inputs:
            self.i_neurons[i].output = input
            i = i + 1
             
    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))
        
    def __feed_forward(self):
        for neuron in self.i_neurons:
            for connection in neuron.connections:
                 connection.neuron.input += neuron.output * connection.weight
        
        for layer in self.hidden_neurons:
            bias_neuron = True
            
            for neuron in layer:
                if bias_neuron:
                    neuron.output = 1
                    bias_neuron = False
                else:
                    neuron.output = self.sigmoid(neuron.input)
                
                for connection in neuron.connections:
                    connection.neuron.input += neuron.output * connection.weight
                    
        for neuron in self.o_neurons:
            neuron.output = self.sigmoid(neuron.input)
    
    def ff_output(self):
        output = [neuron.output for neuron in self.o_neurons]
        return output
    
    def __set_target_output(self, Output):
        self.Targetoutput = Output
    
    def set_learning_rate(self, n):
        self.learning = n
     
    def cal_delta_outLayer(self,target, output):
        return (target - output) * output * (1-output)    
        
    def __delta_pass(self):
        i = 0
        for neuron in self.o_neurons:
            neuron.delta = self.cal_delta_outLayer(self.Targetoutput[i], neuron.output)
            i = i + 1
        
        num_of_layers = len(self.hidden_neurons)
        for k in range(num_of_layers-1, -1, -1):
            layer = self.hidden_neurons[k]
            #skip delta for bias unit
            for neuron in layer[1:]:
                d = 0
                for connection in neuron.connections:
                    d += connection.neuron.delta * connection.weight
                neuron.delta = neuron.output * (1- neuron.output) * d

        
        
    def __backpropagate(self):
        self.__delta_pass()
        
        layers = [self.i_neurons]
        for layer in self.hidden_neurons:
            layers.append(layer)
        
        for layer in layers:
            for neuron in layer:
                for connection in neuron.connections:
                    connection.dell += connection.neuron.delta * neuron.output
                    
                neuron.input = 0
        
        for neuron in self.o_neurons:
            neuron.input = 0
    
    def __update_weights(self):
        
        layers = [self.i_neurons]
        for layer in self.hidden_neurons:
            layers.append(layer)
        
        m = len(self.input_data)
            
        for layer in layers:
            bias_neuron = True
            for neuron in layer:
                for connection in neuron.connections:
                    if bias_neuron:
                        differential = (connection.dell / m)
                    else:
                        differential = (connection.dell  + self.regularization * connection.weight) / m
                    
                    connection.weight += self.learning * differential
                    connection.dell = 0
                bias_neuron = False
                
            
    def __calculateError(self):
        err = 0
        i  = 0
        for neuron in self.o_neurons:
            err += pow(self.Targetoutput[i] - neuron.output, 2)
            i += 1
        err = err / 2.0
        return err
    
    def __reset_input_inner(self):
        
        for layer in self.hidden_neurons:
            for neuron in layer:
                neuron.input = 0
                
        for neuron in self.o_neurons:
            neuron.input = 0
        
                    
    def Train(self, input_data, output_data, learning_rate, iterations):
        self.learning = learning_rate
        self.input_data = input_data
        self.output_data = output_data
        self.regularization = 0
        
        itrError = []   
        for i in range(iterations):
            errors = []
            m = len(input_data)
            for k in range(m):
                self.__set_input(input_data[k])
                self.__set_target_output(output_data[k])
                self.__feed_forward()
                errors.append(self.__calculateError())
                self.__backpropagate()
            avgError = sum(errors) / m
            itrError.append(avgError)
            self.__update_weights()
            
            print("Iteration: " + str(i+1) + " Error: " + str(avgError))
            
        return itrError
    
    def Test(self, test_input_data, test_output_data):
        m = len(test_input_data)
        errors = []
        for i in range(m):
            self.__set_input(test_input_data[i])
            self.__set_target_output(test_output_data[i])
            self.__feed_forward()
            errors.append(self.__calculateError())
            self.__reset_input_inner()
        avgError = sum(errors) / m
        return avgError
        
    def get_weights(self):
        weights = []
        layers = [self.i_neurons]
        for layer in self.hidden_neurons:
            layers.append(layer)
        
        for layer in layers:
            w_layer = []
            for neuron in layer:
                w_neuron = []
                for connection in neuron.connections:
                    w_neuron.append(connection.weight)
                w_layer.append(w_neuron)
            weights.append(w_layer)
        
        return weights
        
    


                        
   


