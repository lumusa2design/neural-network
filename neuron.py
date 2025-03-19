import numpy as np

class  Neuron:
    def __init__(self, num_inputs):
        self.weight = np.random.randn(num_inputs)
        self.bias = np.random.randn()
        self.output = 0
        self.inputs = None
        self.derivative_weight = np.zeros_like(self.weight)
        self.derivative_bias = 0

    def activation_funciton(self, x):
        return 1/(1+np.exp(-1))

    def derivative_activation(self,x):
        return x* (1-x)

    def forward(self, inputs):
        self.inputs = inputs
        weighted_sum = np.dot(inputs, self.weight) + self.bias
        self.output = self.activation_funciton(weighted_sum)
        return self.output

    def backward(self, d_output,learning_rate):
        d_activation = d_output * self.derivative_activation((self.output))
        self.derivative_weight = np.dot(self.inputs, d_activation)
        self.derivative_bias = d_activation
        d_input = np.dot(d_activation, self.weight)
        self.weight -= self.derivative_weight * learning_rate
        self.bias -= learning_rate * self.derivative_bias
        return d_input

if __name__ == "__main__":
    neuron = Neuron(3)
    inputs = np.array([1,2,3])
    output = neuron.forward(inputs)
    print(output)