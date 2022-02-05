from abc import ABC, abstractmethod

import numpy as np


class Regularizer(ABC):
    @abstractmethod
    def f(self, x: np.ndarray) -> float:
        pass

    @abstractmethod
    def df(self, x: np.ndarray) -> float:
        pass

    # def __call__(self, *args, **kwargs):
    #     return self.f(*args, **kwargs)

    def __str__(self):
        attrs = ', '.join([f'{k}={v}' for k, v in self.__dict__.items() if not k.startswith('_')])
        return f'{self.__class__.__name__}({attrs})'


class L1(Regularizer):
    def __init__(self, l1: float):
        self.l1 = l1

    def f(self, x: np.ndarray) -> float:
        return self.l1 * np.abs(x)

    def df(self, x: np.ndarray) -> float:
        return self.l1 * np.sign(x)

    def __str__(self):
        return f'{self.__class__.__name__}({self.l1})'


class L2(Regularizer):
    def __init__(self, l2: float):
        self.l2 = l2

    def f(self, x: np.ndarray) -> float:
        return self.l2 * 0.5 * np.square(x)

    def df(self, x: np.ndarray) -> float:
        return self.l2 * x

    def __str__(self):
        return f'{self.__class__.__name__}({self.l2})'


class L1L2(Regularizer):
    def __init__(self, l1: float, l2: float):
        self.l1 = l1
        self.l2 = l2
        self._l1_regul = L1(l1)
        self._l2_regul = L2(l2)

    def f(self, x: np.ndarray) -> float:
        return self._l1_regul.f(x) + self._l2_regul.f(x)

    def df(self, x: np.ndarray) -> float:
        return self._l1_regul.df(x) + self._l2_regul.df(x)
