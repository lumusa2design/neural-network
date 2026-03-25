import numpy as np
from layer import Layer
import json
class NeuralNetwork:
    def __init__(self):
        self.layers = []
        self.loss_list = []
            
    def add_layer(self, num_neurons, input_size=None, activation="sigmoid"):
        if not self.layers:
            if input_size is None:
                raise ValueError("La primera capa necesita input_size")
            self.layers.append(Layer(num_neurons, input_size, activation))
        else:
            previous_output_size = len(self.layers[-1].neurons)
            self.layers.append(Layer(num_neurons, previous_output_size, activation))

    def forward(self, inputs):
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs

    def backward(self, loss_gradient, learning_rate):
        for layer in reversed(self.layers):
            loss_gradient = layer.backward(loss_gradient, learning_rate)

    def train(self, X, Y, epochs=1000, learning_rate=0.1):
        for epoch in range(epochs):
            loss = 0
            for i in range(len(X)):
                output = self.forward(X[i])
                loss += np.mean((Y[i] - output) ** 2)
                loss_gradient = 2 * (output - Y[i])
                self.backward(loss_gradient, learning_rate)

            loss /= len(X)
            self.loss_list.append(loss)

            if epoch % 100 == 0:
                print(f"Epoch: {epoch}, loss: {loss}")

    def predict(self, X):
        predictions = []
        for i in range(len(X)):
            predictions.append(self.forward(X[i]))
        return np.array(predictions)
    
    def save_model(self, file_path):
        model_data = {
            "layers": [layer.to_dict() for layer in self.layers]
        }
        with open(file_path, 'w') as f:
            json.dump(model_data, f)


if __name__ == "__main__":
    X = np.array([
        [0.5, 0.2, 0.1],
        [0.9, 0.7, 0.3],
        [0.4, 0.6, 0.8]
    ])

    y = np.array([
        [0.3],
        [0.6],
        [0.9]
    ])

if __name__ == "__main__":
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ], dtype=float)

    y = np.array([
        [0],
        [1],
        [1],
        [0]
    ], dtype=float)

    nn = NeuralNetwork()
    nn.add_layer(num_neurons=4, input_size=2, activation="relu")
    nn.add_layer(num_neurons=1, activation="sigmoid")

    nn.train(X, y, epochs=5000, learning_rate=0.1)

    predictions = nn.predict(X)
    print("Predicciones:")
    print(predictions)
    print("Valores reales:")
    print(y)