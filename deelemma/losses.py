from abc import abstractmethod, ABC

import numpy as np


class Loss(ABC):
    @staticmethod
    @abstractmethod
    def _f(y_hat: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Loss function"""
        pass

    @staticmethod
    @abstractmethod
    def _df(y_hat: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Loss function derivative with respect to y_hat"""
        pass

    def _total_loss(self, y_hat: np.ndarray, t: np.ndarray) -> float:
        # Sum in axis=1

        loss_n = np.sum(self._f(y_hat, t), axis=1)  # dim=(N, F)->(N,)
        # loss_n = self._f(y_hat, t)

        # Mean value in axis=0
        return float(np.mean(loss_n))  # dim=(N,)->scalar

    @property
    def f(self):
        return self._total_loss

    @property
    def df(self):
        return self._df

    # def __call__(self, *args, **kwargs):
    #     return self.f(*args, **kwargs)

    def __str__(self):
        attrs = ', '.join([f'{k}={v}' for k, v in self.__dict__.items() if not k.startswith('_')])
        return f'{self.__class__.__name__}({attrs})'


class MAE(Loss):
    @staticmethod
    def _f(y_hat: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.abs(y_hat - y)

    @staticmethod
    def _df(y_hat: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.sign(y_hat - y)


class MSE(Loss):
    @staticmethod
    def _f(y_hat: np.ndarray, y: np.ndarray) -> np.ndarray:
        return 0.5 * np.square(y_hat - y)

    @staticmethod
    def _df(y_hat: np.ndarray, y: np.ndarray) -> np.ndarray:
        return y_hat - y


class BinaryCrossentropy(Loss):
    @staticmethod
    def _f(y_hat: np.ndarray, y: np.ndarray) -> np.ndarray:
        return - y * np.log(y_hat) - (1 - y) * np.log(1 - y_hat)

    @staticmethod
    def _df(y_hat: np.ndarray, y: np.ndarray) -> np.ndarray:
        return - y / y_hat - (1 - y) / (1 - y_hat)


class CategoricalCrossentropy(Loss):  # TODO categorical_crossentropy increases, does not work
    @staticmethod
    def _f(y_hat: np.ndarray, y: np.ndarray) -> np.ndarray:
        return - y * np.log(y_hat + 1e-9)

    @staticmethod
    def _df(y_hat: np.ndarray, y: np.ndarray) -> np.ndarray:
        return - y / (y_hat + 1e-9)


if __name__ == '__main__':
    cost = MSE()
    print(cost)
