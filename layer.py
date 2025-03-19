from neuron import Neuron
import numpy as np

class Layer:
    def __init__(self, num_neurons, input_size):
        self.neurons = [Neuron(input_size) for _ in range (num_neurons)]

    def forward(self, inputs):
        return np.array([neuron.forward(inputs) for neuron in self.neurons])

    def backward(self, d_outputs, learning_rate):
        d_inputs = np.zeros(self.neurons[0].inputs)
        for i, neuron in enumerate(self.neurons):
            d_inputs += neuron.backward(d_outputs[i], learning_rate)
        return d_inputs

if __name__ == "__main__":
    layer = Layer(3,4)
    inputs = np.array([1,8,5,6])

    layer_output = layer.forward(inputs)
    print("Layer outputs: ", layer_output)


