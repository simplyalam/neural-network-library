import numpy as np
import pandas as pd

class NeuralNet():
    def __init__(self, input_size, hidden_size, output_size, num):
        np.random.seed(458)

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num = num

        self.input = np.array([1] * self.input_size)
        self.output = np.array([1] * self.output_size)

        self.weights = []
        self.h_layers = []

        for i in range(0,self.num + 1):
            if (i == 0):
                self.weights.append(np.random.randn(self.input_size, self.hidden_size))
            elif (i == num):
                self.weights.append(np.random.randn(self.hidden_size, self.output_size))
            else:
                self.weights.append(np.random.randn(self.hidden_size, self.hidden_size))

        temp_layer = self.input
        for i in range(self.num):
            if (i == 0):
                temp_layer = self.sigmoid(np.dot(self.input, self.weights[i]))
            else:
                temp_layer = self.sigmoid(np.dot(temp_layer, self.weights[i]))
            self.h_layers.append(temp_layer)

    @staticmethod
    def sigmoid(x):
        return 1/(1 + np.exp(-x))

    @staticmethod
    def sigmoid_prime(x):
        return NeuralNet.sigmoid(x) * (1 - NeuralNet.sigmoid(x))

    @staticmethod
    def tanh(x):
        return 2 * NeuralNet.sigmoid(2 * x) - 1

    @staticmethod
    def tanh_prime(x):
        return 1 - NeuralNet.tanh(x) * NeuralNet.tanh(x)

    def set_input(self, input):
        self.input = input

    def forward(self):

        temp_layer = self.input
        for i in range(self.num):
            if (i == 0):
                temp_layer = self.sigmoid(np.dot(self.input, self.weights[i]))
            else:
                temp_layer = self.sigmoid(np.dot(temp_layer, self.weights[i]))
            self.h_layers[i] = temp_layer

        self.output = self.sigmoid(np.dot(temp_layer, self.weights[i + 1]))

    def back(self, y):
        y_err = y - self.output
        y_delta = y_err * self.sigmoid_prime(self.output)

        temp_delta = y_delta
        for i in range(self.num - 1,-1,-1):

            hidden_err = temp_delta.dot(self.weights[i + 1].T)
            hidden_delta = hidden_err * self.sigmoid_prime(self.h_layers[i])

            self.weights[i + 1] += self.h_layers[i].T.dot(temp_delta)
            temp_delta = hidden_delta

nn = NeuralNet(input_size=800*600*2, hidden_size=25, output_size=2, num=5)

raw = np.genfromtxt("data.csv", delimiter=" ")

y = raw[:, -2:]
y = nn.sigmoid(y)
nn.input = raw[:, :-2]

print("Input:\n" + str(nn.input))

for i in range(2001):
    nn.forward()

    y_err = y - nn.output

    if (i % 1000) == 0:
        print("Error: " + str(np.mean(np.abs(y_err))))

    nn.back(y)
print("Expected output:\n" + np.array_str(y, precision=3))
print("Actual output:\n" + np.array_str(nn.output, precision=3))