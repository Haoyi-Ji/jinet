"""
A nerual net will be made up of layers.
"""
from typing import Dict, Callable
import numpy as np
from jinet.tensor import Tensor


class Layer:
    def __init__(self) -> None:
        self.params: Dict[str, Tensor] = {}
        self.grads: Dict[str, Tensor] = {}

    def forward(self, inputs: Tensor) -> Tensor:
        raise NotImplementedError

    def backward(self, grad: Tensor) -> Tensor:
        raise NotImplementedError



class Linear(Layer):
    """
    output = input * w + b
    """
    def __init__(self, input_size: int, output_size: int) -> None:
        super().__init__()
        self.params['w'] = np.random.randn(input_size, output_size)
        self.params['b'] = np.random.randn(output_size)

    def forward(self, inputs: Tensor) -> Tensor:
        self.inputs = inputs
        return inputs @ self.params['w'] + self.params['b']

    def backward(self, grad: Tensor) -> Tensor:
        """
        if y = f(x) and x = a @ b + c
        then dy/da = f'(x) @ b.T
        and dy/db = a.T @ f'(x)
        and dy/dc = f'(x) 
        """
        self.grads['b'] = np.sum(grad, axis=0)
        self.grads['w'] = self.inputs.T @ grad
        return grad @ self.params['w'].T


F = Callable[[Tensor], Tensor]


class Activation(Layer):
    '''
    An activation layer just applies a function elementwise to its inputs
    '''
    def __init__(self, f: F, f_prime: F) -> None:
        super().__init__()
        self.f = f
        self.f_prime = f_prime

    def forward(self, inputs: Tensor) -> Tensor:
        self.inputs = inputs
        return self.f(inputs)

    def backward(self, grad: Tensor) -> Tensor:
        '''
        if y = f(x) and x = g(z)
        then dy/dz = f'(x) * g'(z)
        '''
        return self.f_prime(self.inputs) * grad



# tanh
def tanh(x: Tensor) -> Tensor:
    return np.tanh(x)


def tanh_prime(x: Tensor) -> Tensor:
    y = tanh(x)
    return 1 - y ** 2


class Tanh(Activation):
    def __init__(self):
        super().__init__(tanh, tanh_prime)


# sigmoid
def sigmoid(x: Tensor) -> Tensor:
    return 1 / (1 + np.exp(-x))


def sigmoid_prime(x: Tensor) -> Tensor:
    y = sigmoid(x)
    return y * (1 - y)


class Sigmoid(Activation):
    def __init__(self):
        super().__init__(sigmoid, sigmoid_prime)


# relu
def relu(x: Tensor) -> Tensor:
    return np.maximum(x, 0, x)


def relu_prime(x: Tensor) -> Tensor:
    return 1. * (x > 0)


class Relu(Activation):
    def __init__(self):
        super().__init__(relu, relu_prime)



# softmax
def softmax(x: Tensor) -> Tensor:
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def softmax_prime(x: Tensor) -> Tensor:
    x = x - np.max(x)
    sigma = np.exp(x).sum(axis=0)
    return (np.exp(x) * sigma - np.exp(x) ** 2) / sigma ** 2


class Softmax(Activation):
    def __init__(self):
        super().__init__(softmax, softmax_prime)


# leaky relu
def leaky_relu(x: Tensor, alpha: float = 0.01):
    return (1. * (x >= 0) - alpha * (x < 0)) * x


def leaky_relu_prime(x: Tensor, alpha: float = 0.01):
    return 1. * (x >= 0) - alpha * (x < 0)


class LeakyRelu(Activation):
    def __init__(self):
        super().__init__(leaky_relu, leaky_relu_prime)


# ELU
def elu(x: Tensor, alpha: float = 0.01) -> Tensor:
    return x * (x >= 0) - alpha * (np.exp(x) - 1) * (x < 0)


def elu_prime(x: Tensor, alpha: float = 0.01) -> Tensor:
    y = elu(x, alpha)
    return x * (x >= 0) - (y + alpha) * (x < 0)


class Elu(Activation):
    def __init__(self):
        super().__init__(elu, elu_prime)



# softplus
def softplus(x: Tensor) -> Tensor:
    return np.log(1 + np.exp(x))


def softplus_prime(x: Tensor) -> Tensor:
    return 1 / (1 + np.exp(-x))


class SoftPlus(Activation):
    def __init__(self):
        super().__init__(softplus, softplus_prime)