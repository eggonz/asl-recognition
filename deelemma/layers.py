from typing import Tuple, List, Optional

import numpy as np

from .activation import Activation
from .base_layer import Layer, TrainableLayer
from .parameters import ParameterInitializer, ParameterTensor
from .regularizers import Regularizer
from .utils.exceptions import InvalidShapeException
from .utils.generic import conv2d, ConvPadding, rot180, maxpool2d


class DenseLayer(TrainableLayer):
    def __init__(self, input_shape: Tuple[int], output_shape: Tuple[int],
                 kernel_initializer: ParameterInitializer = ParameterInitializer.NOISE,
                 bias_initializer: ParameterInitializer = ParameterInitializer.ZEROS,
                 kernel_regularizer: Optional[Regularizer] = None,
                 bias_regularizer: Optional[Regularizer] = None):
        if len(input_shape) != 1 or len(output_shape) != 1:
            raise InvalidShapeException('Input and output shape not flat')
        super(DenseLayer, self).__init__(input_shape, output_shape, kernel_regularizer, bias_regularizer)

        self.weights = ParameterTensor(shape=(self.input_shape[1], self.output_shape[1]), param_init=kernel_initializer)
        self.bias = ParameterTensor(shape=(1, self.output_shape[1]), param_init=bias_initializer)

    def _forward(self, x: np.ndarray) -> np.ndarray:
        return x @ self.weights.value + self.bias.value

    def _backward(self, error: np.ndarray) -> np.ndarray:
        return error @ self.weights.value.T

    def _compute_grad(self) -> None:
        samples = self._input.shape[0]
        self.weights.grad = (1.0 / samples) * self._input.T @ self._error
        self.bias.grad = np.mean(self._error, axis=0, keepdims=True)
        if self.kernel_regularizer is not None:
            self.weights.grad = self.weights.grad + self.kernel_regularizer.df(self.weights.value)
        if self.bias_regularizer is not None:
            self.bias.grad = self.bias.grad + self.bias_regularizer.df(self.bias.value)

    def _inference(self, x: np.ndarray) -> np.ndarray:
        return x @ self.weights.value + self.bias.value

    def get_parameters(self) -> List[ParameterTensor]:
        return [self.weights, self.bias]


class ActivationLayer(Layer):
    def __init__(self, shape: tuple, act: Activation):
        super(ActivationLayer, self).__init__(shape, shape)

        self.act = act

    def _forward(self, x: np.ndarray) -> np.ndarray:
        return self.act.f(x)

    def _backward(self, error: np.ndarray) -> np.ndarray:
        if self.act.df_is_tensor():
            return np.tensordot(error, self.act.df(self._input))
        return np.multiply(error, self.act.df(self._input))

    def _inference(self, x: np.ndarray) -> np.ndarray:
        return self.act.f(x)


class DropoutLayer(Layer):
    def __init__(self, shape: tuple, dropout_rate: float):
        super(DropoutLayer, self).__init__(shape, shape)

        self.dropout_rate = dropout_rate
        self._dropped = None

    def on_iteration_start(self) -> None:
        super(DropoutLayer, self).on_iteration_start()
        self._dropped = (np.random.rand(*self.input_shape[1:]) > self.dropout_rate).astype(float)

    def _forward(self, x: np.ndarray) -> np.ndarray:
        if self._dropped is None:
            raise Exception('Variable referenced before assignment')
        return np.multiply(x, self._dropped)

    def _backward(self, error: np.ndarray) -> np.ndarray:
        if self._dropped is None:
            raise Exception('Variable referenced before assignment')
        return np.multiply(error, self._dropped)

    def _inference(self, x: np.ndarray) -> np.ndarray:
        # Dropout does not apply in inference
        return x


class FlattenLayer(Layer):
    def __init__(self, input_shape: tuple):
        super(FlattenLayer, self).__init__(input_shape, (np.prod(input_shape),))

    def _forward(self, x: np.ndarray) -> np.ndarray:
        return x.reshape(self.output_shape)

    def _backward(self, error: np.ndarray) -> np.ndarray:
        return error.reshape(self.input_shape)

    def _inference(self, x: np.ndarray) -> np.ndarray:
        return x.reshape(self.output_shape)


class Conv2dLayer(TrainableLayer):
    def __init__(self, input_shape: Tuple[int, int, int], filters: int, kernel_shape: Tuple[int, int],
                 kernel_initializer: ParameterInitializer = ParameterInitializer.NOISE,
                 kernel_regularizer: Optional[Regularizer] = None):
        if len(input_shape) != 3:
            raise InvalidShapeException('Invalid input shape')
        if len(kernel_shape) != 2:
            raise InvalidShapeException('Invalid kernel shape')
        h, w, ch = input_shape
        kh, kw = kernel_shape
        if h - kh < 0 or w - kw < 0:
            raise InvalidShapeException('Larger convolution kernel cannot be applied over smaller input')
        super(Conv2dLayer, self).__init__(input_shape, (h - kh + 1, w - kw + 1, filters),
                                          kernel_regularizer, None)

        self._filters = filters
        self._kernel_shape = kernel_shape

        # TODO bias
        # layer without bias (commonly one additional bias term per filter, shape=(1,1,1,f))
        self.kernel_tensor = ParameterTensor(shape=(*kernel_shape, ch, filters), param_init=kernel_initializer)

    def _forward(self, x: np.ndarray) -> np.ndarray:
        return conv2d(x, self.kernel_tensor.value, ConvPadding.VALID)

    def _backward(self, error: np.ndarray) -> np.ndarray:
        kernel_flipped = rot180(self.kernel_tensor.value)
        kernel_flipped.swapaxes(2, 3)
        return conv2d(error, kernel_flipped.swapaxes(2, 3), ConvPadding.FULL)

    def _inference(self, x: np.ndarray) -> np.ndarray:
        return conv2d(x, self.kernel_tensor.value, ConvPadding.VALID)

    def _compute_grad(self) -> None:
        x_permut = np.transpose(self._input, (3, 1, 2, 0))
        error_permut = np.transpose(self._error, (1, 2, 0, 3))
        grad_permut = conv2d(x_permut, error_permut, ConvPadding.VALID)
        grad = np.transpose(grad_permut, (1, 2, 0, 3))
        if self.kernel_regularizer is not None:
            self.kernel_tensor.grad = grad + self.kernel_regularizer.df(self.kernel_tensor.value)
        else:
            self.kernel_tensor.grad = grad

    def get_parameters(self) -> List[ParameterTensor]:
        return [self.kernel_tensor]


class MaxPool2dLayer(Layer):
    def __init__(self, input_shape: Tuple[int, int, int], kernel_shape: Tuple[int, int]):
        h, w, c = input_shape
        kh, kw = kernel_shape
        if h < kh or w < kw:
            raise InvalidShapeException('Larger pooling kernel cannot be applied over smaller input')
        super(MaxPool2dLayer, self).__init__(input_shape, output_shape=(h // kh, w // kw, c))

        self.kernel_shape = kernel_shape
        self._max_loc = None

    def _forward(self, x: np.ndarray) -> np.ndarray:
        output_tensor = maxpool2d(x, self.kernel_shape)

        h, w = self.input_shape[1:3]
        kh, kw = self.kernel_shape
        yh, yw = self.output_shape[1:3]

        input_cut = self._input[:, 0:kh * yh, 0:kw * yw, :]  # (kh*yh,kw*yw)
        out_repeat = output_tensor.repeat(kh, axis=1).repeat(kw, axis=2)  # (kh*yh,kw*yw)
        max_loc_cut = np.equal(input_cut, out_repeat).astype(int)  # (kh*yh,kw*yw)
        self._max_loc = np.pad(max_loc_cut, ((0, 0), (0, h - kh * yh), (0, w - kw * yw), (0, 0)),
                               mode='constant', constant_values=0)  # (kh*yh+h-kh*yh,kw*yw+w-kw*yw)=(h,w)
        return output_tensor

    def _backward(self, error: np.ndarray) -> np.ndarray:
        h, w = self.input_shape[1:3]
        kh, kw = self.kernel_shape
        yh, yw = self.output_shape[1:3]

        err_repeat = error.repeat(2, axis=1).repeat(2, axis=2)  # (kh*yh,kw*yw)
        err_repeat_pad = np.pad(err_repeat, ((0, 0), (0, h - kh * yh), (0, w - kw * yw), (0, 0)),
                                mode='constant', constant_values=0)  # (h,w)
        return np.multiply(err_repeat_pad, self._max_loc)  # (h,w)

    def _inference(self, x: np.ndarray) -> np.ndarray:
        return maxpool2d(x, self.kernel_shape)


class BatchNormLayer(Layer):
    def __init__(self, shape: tuple, epsilon: int = 1e-3):
        super().__init__(shape, shape)
        self.epsilon = epsilon

        # TODO use trainable parameters
        self._gamma = np.ones(shape)
        self._beta = np.zeros(shape)

        self._batch_means = []
        self._batch_vars = []
        self._inf_mean = 0
        self._inf_var = 1

        self._x_hat = None
        self._mean = None
        self._var = None

    def on_training_start(self) -> None:
        super(BatchNormLayer, self).on_training_start()
        self._batch_means = []
        self._batch_vars = []
        self._inf_mean = 0
        self._inf_var = 1

    def on_iteration_start(self) -> None:
        super(BatchNormLayer, self).on_iteration_start()
        self._x_hat = None
        self._mean = None
        self._var = None

    def on_training_end(self) -> None:
        super(BatchNormLayer, self).on_training_end()
        self._inf_mean = np.mean(self._batch_means, axis=0)
        self._inf_var = np.mean(self._batch_vars, axis=0)
        self._batch_means = []
        self._batch_vars = []

    def _forward(self, x: np.ndarray) -> np.ndarray:
        self._mean = np.mean(x, axis=0)
        self._var = np.mean((x - self._mean) ** 2, axis=0)
        m = self._input.shape[0]  # batch size
        self._batch_means.append(self._mean)
        self._batch_vars.append(m / (m - 1) * self._var)
        self._x_hat = (x - self._mean) / np.sqrt(self._var + self.epsilon)
        return self._gamma * self._x_hat + self._beta

    def _backward(self, error: np.ndarray) -> np.ndarray:
        d_x_hat = error * self._gamma
        # d_gamma = np.sum(error * self._x_hat, axis=0)  # TODO trainable batchnorm parameters
        # d_beta = np.sum(error, axis=0)
        d_var = np.sum(d_x_hat * (self._input - self._mean) *
                       (-0.5) * np.power(self._var + self.epsilon, -3 / 2), axis=0)
        d_mean = np.sum(-d_x_hat * np.power(self._var + self.epsilon, -1 / 2), axis=0) + \
            d_var * (-0.5) * np.mean(self._input - self._mean, axis=0)
        m = self._input.shape[0]  # batch size
        d_x = d_x_hat + np.power(self._var + self.epsilon, -1 / 2) + \
            d_var * 2 / m * (self._input - self._mean) + d_mean / m
        return d_x

    def _inference(self, x: np.ndarray) -> np.ndarray:
        g = self._gamma * np.power(self._inf_var + self.epsilon, -1 / 2)
        b = self._beta - g * self._inf_mean
        return g * x + b
