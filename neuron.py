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
        return 1/(1+np.exp(-x))
    
    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return (x > 0).astype(float)

    def derivative_activation(self,x):
        return x* (1-x)

    def forward(self, inputs):
        self.inputs = inputs
        weighted_sum = np.dot(inputs, self.weight) + self.bias
        self.output = self.activation_funciton(weighted_sum)
        return self.output

    def backward(self, d_output,learning_rate):
        d_activation = d_output * self.derivative_activation((self.output))
        self.derivative_weight = self.inputs * d_activation
        self.derivative_bias = d_activation
        d_input = self.weight * d_activation
        self.weight -= self.derivative_weight * learning_rate
        self.bias -= learning_rate * self.derivative_bias
        return d_input
    
    def to_dict(self):
        return {
            "weight": self.weight.tolist(),
            "bias": self.bias,
            "output": self.output
        }
        
    def from_dict(self, data):
        self.weight = np.array(data["weight"])
        self.bias = data["bias"]
        self.output = data["output"]
    
    def mse(y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    def mse_derivative(y_true, y_pred):
        return 2 * (y_pred - y_true) / y_true.size
    
    def binary_cross_entropy(y_true, y_pred):
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

if __name__ == "__main__":
    neuron = Neuron(3)
    inputs = np.array([1,2,3])
    output = neuron.forward(inputs)
    print(output)