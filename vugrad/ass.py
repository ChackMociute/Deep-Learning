import numpy as np
from vugrad.ops import Normalize
from vugrad.core import Op, MatrixMultiply

class Normalize1(Op):
    @staticmethod
    def forward(context, x):
        sumd = x.sum(axis=1, keepdims=True)
        context['x'], context['sumd'] = x, sumd
        return x / sumd

    @staticmethod
    def backward(context, go):
        x, sumd = context['x'], context['sumd']
        return (go / sumd) - (x/(sumd * sumd)) - go.sum(axis=1, keepdims=True)

cntxt = dict()
x = np.arange(20).reshape(4, -1)
b = np.random.randn(5, 1)
c = np.random.randn(1, 4)

Normalize.forward(cntxt, x)
Normalize.backward(cntxt, np.arange(20).reshape(4, -1))