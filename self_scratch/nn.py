import numpy as np
import functions as F
import layers as L
from collections import OrderedDict


class SimpleNN:

    def __init__(self, input_size, hidden_size, output_size, init_weight_std=0.01):
        self.params = dict()
        self.params["W1"] = init_weight_std * np.random.rand(input_size, hidden_size)
        self.params["b1"] = np.zeros(hidden_size)
        self.params["W2"] = init_weight_std * np.random.rand(hidden_size, output_size)
        self.params["b2"] = np.zeros(output_size)

        self.layers = OrderedDict()
        self.layers["Affine1"] = L.Affine(self.params["W1"], self.params["b1"])
        self.layers["Activation1"] = L.Relu()
        self.layers["Affine2"] = L.Affine(self.params["W2"], self.params["b2"])
        self.layers["Activation2"] = L.Relu()

        self.output_layer = L.Softmax_with_loss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    def loss(self, x, t):
        y = self.predict(x)
        loss = self.output_layer.forward(y, t)
        # print(self.output_layer.y)
        return loss

    def back_propagation(self, x, t):
        self.loss(x, t)

        dout = 1
        dx = self.output_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dx = layer.backward(dx)

        grad = dict()
        grad["W1"] = self.layers["Affine1"].dW
        grad["b1"] = self.layers["Affine1"].db
        grad["W2"] = self.layers["Affine2"].dW
        grad["b2"] = self.layers["Affine2"].db

        return grad

    def pram_update(self, grad, lr):
        for key in ["W1", "b1", "W2", "b2"]:
            self.params[key] -= lr * grad[key]

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)
        accuracy = np.sum(y == t) / float(x.shape[0])

        return accuracy

class SimpleCNN:

    def __init__(self, input_size, output_size, init_weight_std=0.01, filter_num=5, filter_size=3 ,pool_size = 2):
        # input = [C, h, w]
        size = input_size[1]
        conv_out_size = (size - filter_size + 1)
        pool_out_size = int((conv_out_size/2)**2*filter_num)


        self.params = dict()
        self.params["W1"] = init_weight_std * np.random.rand(filter_num,input_size[0], filter_size, filter_size)
        self.params["b1"] = np.zeros(filter_num)
        self.params["W2"] = init_weight_std * np.random.rand(pool_out_size, output_size)
        self.params["b2"] = np.zeros(output_size)

        self.layers = OrderedDict()
        self.layers["Conv"] = L.Convolution(self.params["W1"], self.params["b1"])
        self.layers["Activation1"] = L.Relu()
        self.layers["Pooling"] = L.Pooling(pool_size,pool_size,stride=2)
        self.layers["Affine1"] = L.Affine(self.params["W2"], self.params["b2"])
        self.layers["Activation2"] = L.Relu()

        self.output_layer = L.Softmax_with_loss()
        self.y = None

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    def loss(self, x, t):
        y = self.predict(x)
        loss = self.output_layer.forward(y, t)
        # print(self.output_layer.y)
        return loss

    def back_propagation(self, x, t):

        self.loss(x, t)

        dout = 1
        dx = self.output_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dx = layer.backward(dx)

        grad = dict()
        grad["W1"] = self.layers["Conv"].dW
        grad["b1"] = self.layers["Conv"].db
        grad["W2"] = self.layers["Affine1"].dW
        grad["b2"] = self.layers["Affine1"].db

        return grad

    def pram_update(self, grad, lr):
        for key in ["W1", "b1", "W2", "b2"]:
            self.params[key] -= lr * grad[key]

    def accuracy(self,x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)
        accuracy = np.sum(y == t) / float(y.shape[0])

        return accuracy