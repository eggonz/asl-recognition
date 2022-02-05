from enum import Enum
from typing import TypeVar

import numpy as np

from utils.exceptions import InvalidShapeException


class ParameterInitializer(Enum):
    ZEROS = 'ZEROS'
    RANDOM = 'RANDOM'
    NOISE = 'NOISE'
    NOISE_2 = 'NOISE_2'
    NOISE_4 = 'NOISE_4'

    def get_tensor(self, shape: tuple) -> np.ndarray:
        if self == ParameterInitializer.ZEROS:
            return np.zeros(shape=shape, dtype=float)
        if self == ParameterInitializer.RANDOM:
            return 2 * np.random.rand(*shape) - 1
        if self == ParameterInitializer.NOISE:
            return (2 * np.random.rand(*shape) - 1) * 0.1
        if self == ParameterInitializer.NOISE_2:
            return (2 * np.random.rand(*shape) - 1) * 0.01
        if self == ParameterInitializer.NOISE_4:
            return (2 * np.random.rand(*shape) - 1) * 1e-4


class ParameterTensor:
    def __init__(self, shape: tuple, param_init: ParameterInitializer = ParameterInitializer.ZEROS):
        self._shape = shape
        self._value = param_init.get_tensor(shape)
        self._momentum = param_init.get_tensor(shape)
        self._grad = ParameterInitializer.ZEROS.get_tensor(shape)

    @property
    def shape(self):
        return self._shape

    @property
    def value(self) -> np.ndarray:
        return self._value

    @value.setter
    def value(self, value: np.ndarray) -> None:
        if value.shape != self.shape:
            raise InvalidShapeException('Invalid parameter tensor shape')
        self._value = value

    @property
    def momentum(self) -> np.ndarray:
        return self._momentum

    @momentum.setter
    def momentum(self, momentum: np.ndarray) -> None:
        if momentum.shape != self.shape:
            raise InvalidShapeException('Invalid parameter tensor shape')
        self._momentum = momentum

    @property
    def grad(self) -> np.ndarray:
        return self._grad

    @grad.setter
    def grad(self, grad: np.ndarray) -> None:
        if grad.shape != self.shape:
            raise InvalidShapeException('Invalid parameter tensor shape')
        self._grad = grad

    def reset_value(self, param_init: ParameterInitializer = ParameterInitializer.ZEROS):
        self._value = param_init.get_tensor(self._shape)

    def reset_momentum(self, param_init: ParameterInitializer = ParameterInitializer.ZEROS):
        self._momentum = param_init.get_tensor(self._shape)

    def zero_grad(self):
        self.grad = ParameterInitializer.ZEROS.get_tensor(self._shape)

    def as_vector(self) -> np.ndarray:
        """Transforms parameters to a 1-dimensional vector
        :return: flattened vector of all the trainable parameters
        """
        return np.concatenate([self._value.flatten(), self._momentum.flatten()])

    ParameterTensor = TypeVar('ParameterTensor')

    @classmethod
    def from_vector(cls, vector: np.ndarray, shape: tuple) -> ParameterTensor:
        """Loads parameters from a 1-dimensional vector
        :param vector: flattened vector of all the trainable parameters
        :param shape: shape of the parameter tensor
        """
        param = cls(shape)
        split = np.prod(shape)
        param.value = vector[:split].reshape(shape)
        param.momentum = vector[split:].reshape(shape)
        return param

    def __str__(self) -> str:
        return f'{self.__class__.__name__}()'

    def __len__(self) -> int:
        return int(np.prod(self._shape))
