from NN import NeuralNetwork

import matplotlib.pyplot as plt

def display(errors):
    itr = len(errors)
    plt.scatter(range(1, itr+1), errors)
    plt.show()
    
def XOR():
    
    train_input = [[0,0],[0,1],[1,0],[1,1]]
    train_output = [[0],[1],[1],[0]]
    
    learning_rate = 0.05
    number_of_iterations = 100000
    
    number_of_input_neurons = 2
    number_of_output_neurons = 1
    hidden_layer_neurons = [2]
    
    
    myNN = NeuralNetwork(number_of_input_neurons, 
                       number_of_output_neurons, 
                       hidden_layer_neurons)
    
    itrError = myNN.Train(train_input, train_output, 
                        learning_rate, number_of_iterations)
    
   
    display(itrError)
    
XOR()