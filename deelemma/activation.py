from abc import ABC, abstractmethod

import numpy as np


class Activation(ABC):
    @staticmethod
    @abstractmethod
    def f(x: np.ndarray) -> np.ndarray:
        """Activation function"""
        pass

    @staticmethod
    @abstractmethod
    def df(x: np.ndarray) -> np.ndarray:
        """Returns derivative of the output with respect to the input"""
        pass

    @staticmethod
    def df_is_tensor() -> bool:  # TODO return always tensor jacobian
        """Returns true if the derivative may be taken as a tensor"""
        return False

    def __str__(self):
        attrs = ', '.join([f'{k}={v}' for k, v in self.__dict__.items()])
        return f'{self.__class__.__name__}({attrs})'


class Sigmoid(Activation):
    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def f(x: np.ndarray) -> np.ndarray:
        return Sigmoid._sigmoid(x)

    @staticmethod
    def df(x: np.ndarray) -> np.ndarray:
        y = Sigmoid._sigmoid(x)
        return y * (1 - y)


class Relu(Activation):
    @staticmethod
    def f(x: np.ndarray) -> np.ndarray:
        return x * (x > 0)

    @staticmethod
    def df(x: np.ndarray) -> np.ndarray:
        return 1 * (x > 0)


class Tanh(Activation):
    @staticmethod
    def f(x: np.ndarray) -> np.ndarray:
        return np.tanh(x)

    @staticmethod
    def df(x: np.ndarray) -> np.ndarray:
        return 1 - np.square(np.tanh(x))


class Softmax(Activation):
    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        # FIXME explodes to NaN when x>>1
        exp = np.exp(x - np.max(x))
        return exp / (np.sum(exp, axis=1, keepdims=True) + 1e-9)

    @staticmethod
    def f(x: np.ndarray) -> np.ndarray:
        return Softmax._softmax(x)

    @staticmethod
    def df(x: np.ndarray) -> np.ndarray:
        return Softmax._softmax(x) * (1 - Softmax._softmax(x))

    # @staticmethod
    # def df(x: np.ndarray) -> np.ndarray:
    #     """Returns jacobian: dy(n,u)/dx(m,i)->D(n,u,m,i)"""
    #     y = Softmax._softmax(x)  # y=(n,u), x=(m,i)
    #
    #     _all = np.tensordot(y, np.exp(x), axes=0)  # (n,u,m,i)
    #     _diag = np.tensordot(y, (1 - 2 * np.exp(x)), axes=0)  # (n,u,m,i)
    #
    #     _delta_all = np.identity(x.shape[0])  # (n,m)
    #     _delta_all = _delta_all[:, np.newaxis, :, np.newaxis]  # (n,-,m,-)
    #
    #     _delta_diag = np.tensordot(np.identity(x.shape[0]), np.identity(x.shape[1]), axes=0)  # (n,m,u,i)
    #     _delta_diag = _delta_diag.transpose((0, 2, 1, 3))  # (n,u,m,i)
    #
    #     #print((np.multiply(_all, _delta_all) + np.multiply(_diag, _delta_diag))[:2,:,:2,:])
    #     return np.multiply(_all, _delta_all) + np.multiply(_diag, _delta_diag)
    #
    # @staticmethod
    # def df_is_tensor() -> bool:
    #     return True


if __name__ == '__main__':
    act = Sigmoid()
    print(act)
