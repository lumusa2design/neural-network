import numpy as np

class Neuron:
    def __init__(self, num_inputs, activation="sigmoid"):
        self.weight = np.random.randn(num_inputs) * np.sqrt(1 / num_inputs)
        self.bias = 0.0

        self.output = 0.0
        self.inputs = None
        self.z = None

        self.derivative_weight = np.zeros_like(self.weight)
        self.derivative_bias = 0.0

        self.activation = activation

    def activation_function(self, x):
        if self.activation == "sigmoid":
            return 1 / (1 + np.exp(-x))
        elif self.activation == "relu":
            return np.maximum(0, x)
        else:
            raise ValueError(f"Activación no soportada: {self.activation}")

    def derivative_activation(self):
        if self.activation == "sigmoid":
            return self.output * (1 - self.output)
        elif self.activation == "relu":
            return 1.0 if self.z > 0 else 0.0
        else:
            raise ValueError(f"Activación no soportada: {self.activation}")

    def forward(self, inputs):
        self.inputs = inputs
        self.z = np.dot(inputs, self.weight) + self.bias
        self.output = self.activation_function(self.z)
        return self.output

    def backward(self, d_output, learning_rate):
        d_activation = d_output * self.derivative_activation()
        self.derivative_weight = self.inputs * d_activation
        self.derivative_bias = d_activation
        d_input = self.weight * d_activation

        self.weight -= learning_rate * self.derivative_weight
        self.bias -= learning_rate * self.derivative_bias

        return d_input

    def to_dict(self):
        return {
            "weight": self.weight.tolist(),
            "bias": self.bias,
            "output": self.output,
            "activation": self.activation
        }

    def from_dict(self, data):
        self.weight = np.array(data["weight"])
        self.bias = data["bias"]
        self.output = data["output"]
        self.activation = data.get("activation", "sigmoid")


if __name__ == "__main__":
    neuron = Neuron(3, activation="sigmoid")
    inputs = np.array([1, 2, 3])
    output = neuron.forward(inputs)
    print(output)

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def mse_derivative(y_true, y_pred):
    return 2 * (y_pred - y_true) / y_true.size

def binary_cross_entropy(y_true, y_pred):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def binary_cross_entropy_derivative(y_true, y_pred):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return (y_pred - y_true) / (y_pred * (1 - y_pred) * y_true.size)


if __name__ == "__main__":
    neuron = Neuron(3)
    inputs = np.array([1,2,3])
    output = neuron.forward(inputs)
    print(output)