# for jikken

import numpy as np
import sys, os
import functions as F
import layers as L
from dataset import mnist
import nn
np.set_printoptions(threshold=np.inf)

print((1,2) + (3,))


# c = np.array([[3,2,4,5,6],[1,2,3,4,5]])
# print(c)
# print(c.shape)
# a = np.zeros([3,3,4])
img = np.array([[[0,1,2,3],[10,11,12,13],[20,21,22,23],[30,31,32,33]],[[100,101,102,103],[110,111,112,113],[120,121,122,123],[130,131,132,133]]])
# [[100,101,102,103],[110,111,112,113],[120,121,122,123],[130,131,132,133]]])
filter = np.array([[[1,2],[3,4]], [[5,6],[7,8]]])

c = L.Convolution(W=filter, b=np.zeros(filter.shape), stride=1)
print(c.forward(img))



# print(b)
#
# filter_h, filter_w = filter.shape
# stride = 1
#
# N, H, W = b.shape
# print(N)
# out_h = (H - filter_h) // stride + 1
# out_w = (W - filter_w) // stride + 1
#
#
# col = np.zeros((N, filter_h, filter_w, out_h, out_w))
# # col[0,0,:,:] = b[0:5,0:5]
# # print(col)
# for y in range(filter_h):
#     y_max = y + stride * out_h
#     for x in range(filter_w):
#         x_max = x + stride * out_w
#         col[:, y, x, :, :] = b[:,y:y_max:stride, x:x_max:stride]
#         # print(b[y:y_max:stride, x:x_max:stride])
# print(col)
# # print(col.reshape(1,-1).shape)
#
# col = col.transpose(0,3,4,1,2)
# print(col)
#
# col = col.reshape(N*out_h * out_w, -1)
# print(col)
# # return col