import numpy as np

class Optimizer:
    def __init__(self, lr=0.01):
        self.lr = lr
        self.history = {'loss': [], 'x': []}

class GradientDescent(Optimizer):
    def step(self, x, grad):
        return x - self.lr * grad

class Momentum(Optimizer):
    def __init__(self, lr=0.01, beta=0.9):
        super().__init__(lr)
        self.beta = beta
        self.v = None

    def step(self, x, grad):
        if self.v is None:
            self.v = np.zeros_like(x)
        self.v = self.beta * self.v - self.lr * grad
        return x + self.v

class Nesterov(Optimizer):
    def __init__(self, lr=0.01, beta=0.9):
        super().__init__(lr)
        self.beta = beta
        self.v = None

    def step(self, x, grad_func):
        if self.v is None:
            self.v = np.zeros_like(x)
        x_lookahead = x + self.beta * self.v
        grad = grad_func(x_lookahead)
        self.v = self.beta * self.v - self.lr * grad
        return x + self.v

class AdaGrad(Optimizer):
    def __init__(self, lr=0.01, epsilon=1e-8):
        super().__init__(lr)
        self.epsilon = epsilon
        self.s = None
    def step(self, x, grad):
        if self.s is None: self.s = np.zeros_like(x)
        self.s += grad**2
        return x - self.lr * grad / (np.sqrt(self.s) + self.epsilon)

class RMSProp(Optimizer):
    def __init__(self, lr=0.01, gamma=0.9, epsilon=1e-8):
        super().__init__(lr)
        self.gamma = gamma
        self.epsilon = epsilon
        self.s = None
    def step(self, x, grad):
        if self.s is None: self.s = np.zeros_like(x)
        self.s = self.gamma * self.s + (1 - self.gamma) * (grad**2)
        return x - self.lr * grad / (np.sqrt(self.s) + self.epsilon)

class Adam(Optimizer):
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(lr)
        self.beta1, self.beta2, self.epsilon = beta1, beta2, epsilon
        self.m, self.v, self.t = None, None, 0
    def step(self, x, grad):
        if self.m is None:
            self.m, self.v = np.zeros_like(x), np.zeros_like(x)
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grad**2)
        m_hat = self.m / (1 - self.beta1**self.t)
        v_hat = self.v / (1 - self.beta2**self.t)
        return x - self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)

