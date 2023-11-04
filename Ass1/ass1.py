import numpy as np
import matplotlib.pyplot as plt

from math import e, log
from random import normalvariate
from data import load_synth, load_mnist
from tqdm import tqdm

def sigmoid(x):
    return 1 / (1 + np.exp(x))

def softmax(x):
    x = np.exp(x)
    return x/x.sum(axis=1).reshape(-1, 1)

def sigmoid(x):
    return 1 / (1 + np.exp(x))

def softmax(x):
    x = np.exp(x)
    return x/np.expand_dims(x.sum(axis=1), -1)

class VectorFCN:
    def __init__(self, input_size=784, hidden_size=300, output_size=10, lr=1e-3):
        self.lr = lr
        self.W = np.random.randn(hidden_size, input_size)
        self.b = np.random.randn(hidden_size, 1)
        self.V = np.random.randn(output_size, hidden_size)
        self.c = np.random.randn(output_size, 1)
        self.num_classes = output_size
        self.losses = list()
    
    def classify(self, x):
        return np.argmax(softmax(self.V @ sigmoid(self.W @ x + self.b) + self.c))

    def forward(self, inp, target):
        self.data, self.target = inp, target
        self.h = sigmoid(self.W @ inp + self.b)
        self.y = softmax(self.V @ self.h + self.c)
        self.losses.append(-np.log(self.y[np.arange(len(self.y)), target]).mean())

    def backward(self):
        dloss = -1/self.y[np.arange(len(self.y)), self.target]
        do = dloss * self.y.squeeze() *\
            (np.tile(np.arange(self.num_classes), (len(self.y), 1)) == self.target.reshape(-1, 1)) -\
            self.y[np.arange(len(self.y)), self.target]
        dh = do @ self.V * self.h.squeeze() * (1 - self.h.squeeze())
        self.dV = do.reshape(len(do), -1, 1) @ self.h.swapaxes(1, -1)
        self.dW = dh.reshape(len(dh), -1, 1) @ self.data.swapaxes(1, -1)
        self.dc, self.db = np.expand_dims(do, -1), np.expand_dims(dh, -1)
        self.update_weights()
    
    def update_weights(self):
        self.W -= self.lr * self.dW.mean(axis=0)
        self.b -= self.lr * self.db.mean(axis=0)
        self.V -= self.lr * self.dV.mean(axis=0)
        self.c -= self.lr * self.dc.mean(axis=0)

(xtrain, ytrain), (xval, yval), num_cls = load_mnist()
xtrain, xval = xtrain / 255, xval / 255

fcn = VectorFCN()
fcn.forward(xtrain[:16].reshape(16, -1, 1), ytrain[:16])
fcn.backward()