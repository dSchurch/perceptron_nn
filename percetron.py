import numpy as np

class Perceptron: 
    def __init__(self, inputs, weights, bias):
        self.inputs = np.array(inputs)
        self.weights = np.array(weights)
        self.bias = bias
        self.output =  0
        self.calculate_output()

    def set_bias(self, bias):
        self.bias = bias
    
    def set_inputs(self, inputs):
        self.inputs = np.array(inputs)
    
    def set_weights(self, weights):
        self.weights = np.array(weights)

    def calculate_output(self):
        self.output = 1 if (np.dot(self.inputs, self.weights) + self.bias) > 0 else 0 
        return self.output

class NeuronalNetwork:
    def __init__(self, inputs, hiddenLayerNumber):
        self.input_layer = [Perceptron([0], [0], input) for input in inputs]
        print([i.bias for i in self.input_layer])
        self.hidden_layer = [Perceptron([inputNeuron.output for inputNeuron in self.input_layer], list(np.random.randint(-20, 20, len(self.input_layer))), np.random.randint(-10, 10)) for neuron in range(hiddenLayerNumber)]

nn = NeuronalNetwork([1, 0], 4)

print('-----------------------------')
print('HIDDEN LAYER')
for neuron in nn.hidden_layer:
    print(f'Inputs: {neuron.inputs} - Weights: {neuron.weights} - Output {neuron.output}')
print('-----------------------------')
