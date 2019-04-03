# module for definition layers of NN

import numpy as np
import functions as F

class Relu:

    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        return dx


class Sigmoid:

    def __init__(self):
        self.out = None

    def forward(self, x):  # y = 1/(1+exp(-x))
        out = 1 / (1 + np.exp(-x))
        self.out = out

        return out

    def backward(self, dout):  # dx = dout*y(1-y)
        dx = dout * self.out * (1.0 - self.out)
        return dx


class Affine:                 # this layer should have W, b, dW, db as inner values to update wights and biases later.

    def __init__(self, W, b):
        self.x = None
        self.W = W
        self.b = b
        self.dW = None
        self.db = None

    def forward(self, x):
        self.x_shape= x.shape
        # テンソルの場合も考慮して整形 縦がN行（バッチ数）
        x = x.reshape(self.x_shape[0], -1)
        self.x = x
        out = np.dot(x, self.W) + self.b

        return out

    def backward(self, dout):
        # if dout.ndim == 1 or self.x.ndim == 1:
        #     print("check")
        #     dout = dout.reshape(1, dout.size)
        #     self.x = self.x.reshape(1, self.x.size)
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)


        # テンソルの場合を考慮して整形したものをもとに戻す
        dx = dx.reshape(self.x_shape)
        return dx


class Softmax_with_loss():            # this layer should get labels as one-hot vector

    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None


    def forward(self, x, t):
        self.t = t
        self.y = F.soft_max(x)
        self.loss = F.cross_entropy_loss(self.y, self.t)

        return self.loss

    def backward(self, dout=1):
        batch_size = self.y.shape[0]
        dx = (self.y - self.t) / batch_size

        return dx


class Convolution:              # from source of DL from scratch
    def __init__(self, W, b, stride=1):
        self.W = W
        self.b = b
        self.stride = stride

        self.x = None
        self.col = None
        self.col_W = None

        self.dW = None
        self.db = None

    def forward(self, x):
        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape
        out_h = 1 + int((H - FH) / self.stride)
        out_w = 1 + int((W - FW) / self.stride)

        col = F.im2col(x, FH, FW, self.stride, 0)
        col_W = self.W.reshape(FN, -1).T

        out = np.dot(col, col_W) + self.b
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)

        self.x = x
        self.col = col
        self.col_W = col_W

        return out

    def backward(self, dout):
        FN, C, FH, FW = self.W.shape
        dout = dout.transpose(0, 2, 3, 1).reshape(-1, FN)

        self.db = np.sum(dout, axis=0)
        self.dW = np.dot(self.col.T, dout)
        self.dW = self.dW.transpose(1, 0).reshape(FN, C, FH, FW)

        dcol = np.dot(dout, self.col_W.T)
        dx = F.col2im(dcol, self.x.shape, FH, FW, self.stride, 0)

        return dx


class Pooling:           # from source of  DL from scratch
    def __init__(self, pool_h, pool_w, stride=1):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride

        self.x = None
        self.arg_max = None

    def forward(self, x):
        N, C, H, W = x.shape
        out_h = int(1 + (H - self.pool_h) / self.stride)
        out_w = int(1 + (W - self.pool_w) / self.stride)

        col = F.im2col(x, self.pool_h, self.pool_w, self.stride, 0)
        col = col.reshape(-1, self.pool_h * self.pool_w)

        arg_max = np.argmax(col, axis=1)
        out = np.max(col, axis=1)
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

        self.x = x
        self.arg_max = arg_max

        return out

    def backward(self, dout):
        dout = dout.transpose(0, 2, 3, 1)

        pool_size = self.pool_h * self.pool_w
        dmax = np.zeros((dout.size, pool_size))
        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()
        dmax = dmax.reshape(dout.shape + (pool_size,))

        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        dx = F.col2im(dcol, self.x.shape, self.pool_h, self.pool_w, self.stride, 0)

        return dx

# class Convolution():
#
#     def __init__(self,  W, b, stride=1):
#         self.W = W              # filter
#         self.b = b
#         self.stride = stride
#
#
#     def forward(self, x):
#         N, h, w = x.shape
#         fn, fh, fw = self.W.shape
#
#         out_w = (w - fw) // self.stride + 1
#         out_h = (h - fh) // self.stride + 1
#
#         last_x = self.stride * out_w
#         last_y = self.stride * out_h
#
#         feature_map = np.zeros([N, fn, out_h, out_w])
#
#         for i in range(last_y):
#             for j in range(last_x):
#                 img_part = x[:, i:i + fh, j:j + fw]
#                 # print(img_part)
#                 feature_map[:, i, j] = np.sum(img_part[:] * self.W + self.b, axis=(1, 2))
#
#         return feature_map
#
#     def backward(self):
#         pass
