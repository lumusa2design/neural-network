from neuron import Neuron
import numpy as np

class Layer:
    def __init__(self, num_neurons, input_size, activation="sigmoid"):
        self.neurons = [Neuron(input_size, activation=activation) for _ in range(num_neurons)]

    def forward(self, inputs):
        return np.array([neuron.forward(inputs) for neuron in self.neurons])

    def backward(self, d_values, learning_rate):
        d_inputs = np.zeros_like(self.neurons[0].inputs, dtype=float)

        for i, neuron in enumerate(self.neurons):
            d_inputs += neuron.backward(d_values[i], learning_rate)

        return d_inputs

    def to_dict(self):
        return {
            "neurons": [neuron.to_dict() for neuron in self.neurons]
        }

    def from_dict(self, data):
        for neuron_data, neuron in zip(data["neurons"], self.neurons):
            neuron.from_dict(neuron_data)


if __name__ == "__main__":
    layer = Layer(3, 4, activation="relu")
    inputs = np.array([1, 8, 5, 6])
    layer_output = layer.forward(inputs)
    print("Layer outputs:", layer_output)