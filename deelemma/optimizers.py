from typing import List

import numpy as np

from parameters import ParameterTensor


class Optimizer:
    def __init__(self, model_parameters: List[ParameterTensor]):
        self._model_parameters = model_parameters

    def step(self) -> None:
        """Performs step in parameters space and updates all model parameters
        :return: None
        """
        for param_tensor in self._model_parameters:
            self._update_param(param_tensor)

    def reset(self) -> None:
        """Resets the optimization variables"""

    def _update_param(self, param_tensor: ParameterTensor) -> None:
        """Performs step in parameters space and updates the parameter tensor
        :param param_tensor: parameters to update
        :return: None
        """
        pass

    def __str__(self):
        attrs = ', '.join([f'{k}={v}' for k, v in self.__dict__.items() if not k.startswith('_')])
        return f'{self.__class__.__name__}({attrs})'


class SGD(Optimizer):
    def __init__(self, model_parameters: List[ParameterTensor],
                 lr: float, alpha: float = 0.0):
        """Stochastic Gradient Descent
        :param model_parameters: parameters to optimize
        :param lr: learning rate
        :param alpha: momentum parameter
        """
        super(SGD, self).__init__(model_parameters)
        self.lr = lr
        self.alpha = alpha

    def _update_param(self, param_tensor: ParameterTensor) -> None:
        new_momentum = self.alpha * param_tensor.momentum - self.lr * param_tensor.grad
        new_value = param_tensor.value + new_momentum

        param_tensor.value = new_value
        param_tensor.momentum = new_momentum


class RMSProp(Optimizer):
    def __init__(self, model_parameters: List[ParameterTensor],
                 lr: float, beta: float, epsilon: float = 1e-6):
        """Root Mean Square Propagation
        :param model_parameters: parameters to optimize
        :param lr: learning rate
        :param beta: decay rate
        :param epsilon: stability constant
        """
        super(RMSProp, self).__init__(model_parameters)
        self.lr = lr
        self.beta = beta
        self.epsilon = epsilon
        self._accum = []
        self._reset_accum()

    def _reset_accum(self) -> None:
        self._accum = []
        for param_tensor in self._model_parameters:
            self._accum.append(np.zeros(shape=param_tensor.shape))

    def reset(self) -> None:
        self._reset_accum()

    def step(self) -> None:
        # no momentum
        for k, param_tensor in enumerate(self._model_parameters):
            accum = self._accum[k]

            new_accum = self.beta * accum + (1 - self.beta) * np.multiply(param_tensor.grad, param_tensor.grad)
            delta = np.multiply(1 / np.sqrt(new_accum + self.epsilon), param_tensor.grad)
            new_value = param_tensor.value - self.lr * delta

            self._accum[k] = new_accum
            param_tensor.value = new_value


class AdaGrad(Optimizer):
    def __init__(self, model_parameters: List[ParameterTensor],
                 lr: float, epsilon: float = 1e-7):
        """AdaGrad
        :param model_parameters: parameters to optimize
        :param lr: learning rate
        :param epsilon: stability constant
        """
        super(AdaGrad, self).__init__(model_parameters)
        self.lr = lr
        self.epsilon = epsilon
        self._accum = []
        self._reset_accum()

    def _reset_accum(self) -> None:
        self._accum = []
        for param_tensor in self._model_parameters:
            self._accum.append(np.zeros(shape=param_tensor.shape))

    def reset(self) -> None:
        self._reset_accum()

    def step(self) -> None:
        # no momentum
        for k, param_tensor in enumerate(self._model_parameters):
            accum = self._accum[k]

            new_accum = accum + np.multiply(param_tensor.grad, param_tensor.grad)
            delta = np.multiply(1 / (np.sqrt(new_accum) + self.epsilon), param_tensor.grad)
            new_value = param_tensor.value - self.lr * delta

            self._accum[k] = new_accum
            param_tensor.value = new_value
