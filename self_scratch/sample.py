# for AI training project
# sample for making 3 layer NN

import numpy as np
import functions as F
# 2 input,  3 hidden layer nodes, 2 output

a = np.array([1, 2, 3])
t = np.array([1, 2, 3])
print(a.ndim)
print(a.reshape(1, a.size).ndim)

print(F.cross_entropy_loss(a,t))



# class Three_Layer_NN:
#
#     def __init__(self, input_size, hidden_size, output_size):
#         self.W1 = np.random.randn(input_size, hidden_size)
#         self.B1 = np.random.randn(hidden_size)
#         self.W2 = np.random.randn(hidden_size, output_size)
#         self.B2 = np.random.randn(output_size)
#
#
#     def forward(self, x):
#         z1 = np.dot(x, self.W1) + self.B1
#         h1 = relu(z1)
#         z2 = np.dot(h1, self.W2) + self.B2
#         out = soft_max(z2)
#
#         return out
#
#
# nn = Three_Layer_NN(2,3,2)
# x = np.array([[1, 5]])
#
# print(nn.forward(x))




# sample
#
# X = np.array([[0, 1]])  # 3 input
# W1 = np.array([[1, -2, 3], [7, -8, 9]])  # weight of 1st layer
# B1 = np.array([[9, -8, -5]])  # bias
# W2 = np.array([[3, 5], [-1, 0], [-7, 2]])
# B2 = np.array([3, -6])
#
#
# z1 = np.dot(X, W1) + B1
# h1 = relu(z1)
# z2 = np.dot(h1, W2) + B2
# out = relu(z2)
#
#
#
# print(X)
# print(W1)
# print(B1)
# print(z1)
# print(h1)
# print(W2)
# print(B2)
# print(z2)
# print(out)

