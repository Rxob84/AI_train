# number written manually recognition using Mnist
# 2019/02/26

import numpy as np
import functions as F
import pickle

# use mnist module to load dataset
from dataset import mnist


def get_data():
    (x_train,t_train), (x_test, t_test) = mnist.load_mnist(normalize=True)
    return x_test, t_test


def init_network():
    with open("./dataset/sample_weight.pkl", "rb") as f:
        network = pickle.load(f)

    return network

def predict(network, x):
    W1, W2, W3 = network["W1"], network["W2"], network["W3"]
    b1, b2, b3 = network["b1"], network["b2"], network["b3"]

    a1 = np.dot(x, W1) + b1
    h1 = F.sigmoid(a1)
    a2 = np.dot(h1, W2) + b2
    h2 = F.sigmoid(a2)
    a3 = np.dot(h2, W3) + b3
    out = F.soft_max(a3)

    return out


x, t = get_data()
net = init_network()

num_right = 0
for i in range(len(x)):
    y = predict(net, x[i])
    p = np.argmax(y)
    if p == t[i]:
        num_right += 1
accuracy = num_right/len(x)

print("Accracy : {}".format(accuracy))





