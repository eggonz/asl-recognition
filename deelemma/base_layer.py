from abc import abstractmethod
from typing import Union, List, Optional, Any

import numpy as np

from .parameters import ParameterTensor
from .regularizers import Regularizer
from .utils.exceptions import InvalidShapeException


class Layer:
    def __init__(self, input_shape: tuple, output_shape: tuple):
        self.input_shape: tuple = (-1, *input_shape)
        self.output_shape: tuple = (-1, *output_shape)
        self._input: Optional[np.ndarray] = None
        self._output: Optional[np.ndarray] = None
        self._error: Optional[np.ndarray] = None  # back-propagation error at layer output

    def _check_input_shape(self, x: Union[np.ndarray, tuple]) -> None:
        """Checks whether x matches the input shape
        :param x: array to check or shape tuple
        :return: None
        :raises InvalidShapeException: if wrong shape
        """
        if type(x) is not tuple:
            x = x.shape
        if x[1:] != self.input_shape[1:]:
            raise InvalidShapeException('Invalid input shape')

    def _check_output_shape(self, y: Union[np.ndarray, tuple]) -> None:
        """Checks whether y matches the output shape
        :param y: array to check or shape tuple
        :return: None
        :raises InvalidShapeException: if wrong shape
        """
        if type(y) is not tuple:
            y = y.shape
        if y[1:] != self.output_shape[1:]:
            raise InvalidShapeException('Invalid output shape')

    def on_training_start(self) -> None:
        """Method executed when training starts.
        Useful for clearing the variables used at training level."""
        pass

    def on_epoch_start(self) -> None:
        """Method executed when an epoch starts.
        Useful for clearing the variables used at epoch level."""
        pass

    def on_iteration_start(self) -> None:
        """Method executed when an iteration starts.
        Useful for clearing the variables used at batch-iteration level.
        An iteration corresponds to a single forward-backward-optimize loop for a batch."""
        self._input = None
        self._output = None
        self._error = None

    def on_iteration_end(self) -> None:
        """Method executed when an iteration ends."""
        pass

    def on_epoch_end(self) -> None:
        """Method executed when an epoch ends."""
        pass

    def on_training_end(self) -> None:
        """Method executed when training ends."""
        pass

    @abstractmethod
    def _forward(self, x: np.ndarray) -> np.ndarray:
        pass

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Computes forward pass and updates stored layer output
        :param x: layer input
        :return: layer output
        """
        self._check_input_shape(x)
        self._input = x
        self._output = self._forward(x)
        return self._output

    @abstractmethod
    def _backward(self, error: np.ndarray) -> np.ndarray:
        pass

    def backward(self, error: np.ndarray) -> np.ndarray:
        """Back-propagates the layer error (at output) back to the layer input.
        Also updates stored layer error (at output)
        :param error: backpropagation error at layer output
        :return: backpropagation error at layer input
        """
        self._check_output_shape(error)
        self._error = error
        self._compute_grad()
        return self._backward(error)

    def _compute_grad(self) -> None:
        pass

    @abstractmethod
    def _inference(self, x: np.ndarray) -> np.ndarray:
        pass

    def inference(self, x: np.ndarray) -> np.ndarray:
        """Computes and returns the output of the layer for the given input
        :param x: layer input
        :return: layer output
        """
        return self._inference(x)

    def get_parameters(self) -> List[ParameterTensor]:
        """Return the list of the trainable parameter tensors of the layer"""
        return []

    def get_summary_info(self) -> str:
        """Returns layer relevant info for layer summary"""

        def is_valid(key: str, value: Any) -> bool:
            if key.startswith('_') or key.endswith('_shape'):
                return False
            if value is None:
                return False
            if isinstance(value, ParameterTensor):
                return False
            return True

        return ', '.join([f'{k[0]}={str(v)}' for k, v in self.__dict__.items() if is_valid(k, v)])


class TrainableLayer(Layer):
    def __init__(self, input_shape: tuple, output_shape: tuple,
                 kernel_regularizer: Optional[Regularizer] = None,
                 bias_regularizer: Optional[Regularizer] = None):
        super(TrainableLayer, self).__init__(input_shape, output_shape)
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer

    @abstractmethod
    def _compute_grad(self) -> None:
        pass

    @abstractmethod
    def get_parameters(self) -> List[ParameterTensor]:
        pass


class LayerGroup(Layer):
    def __init__(self, layers: List[Layer]):
        self._check_layer_shapes(layers)
        super().__init__(layers[0].input_shape[1:], layers[-1].output_shape[1:])

        self.layers: List[Layer] = layers
        self.depth = len(layers)

    @staticmethod
    def _check_layer_shapes(layers: List[Layer]) -> None:
        """Checks whether the input shape of each layer in the list matches the output shape of the previous layer
        :param layers: list of layers
        :return: None
        :raises InvalidShapeException: if wrong shapes
        """
        for i in range(1, len(layers)):
            layers[i]._check_input_shape(layers[i - 1].output_shape)

    def on_training_start(self) -> None:
        super(LayerGroup, self).on_training_start()
        for layer in self.layers:
            layer.on_training_start()

    def on_epoch_start(self) -> None:
        super(LayerGroup, self).on_epoch_start()
        for layer in self.layers:
            layer.on_epoch_start()

    def on_iteration_start(self) -> None:
        super(LayerGroup, self).on_iteration_start()
        for layer in self.layers:
            layer.on_iteration_start()

    def on_iteration_end(self) -> None:
        super(LayerGroup, self).on_iteration_end()
        for layer in self.layers:
            layer.on_iteration_end()

    def on_epoch_end(self) -> None:
        super(LayerGroup, self).on_epoch_end()
        for layer in self.layers:
            layer.on_epoch_end()

    def on_training_end(self) -> None:
        super(LayerGroup, self).on_training_end()
        for layer in self.layers:
            layer.on_training_end()

    def _forward(self, x: np.ndarray) -> np.ndarray:
        fwd = x
        for layer in self.layers:
            fwd = layer.forward(fwd)
        return fwd

    def _backward(self, error: np.ndarray) -> np.ndarray:
        err = error
        for layer in reversed(self.layers):
            err = layer.backward(err)
        return err

    def _inference(self, x: np.ndarray) -> np.ndarray:
        fwd = x
        for layer in self.layers:
            fwd = layer.inference(fwd)
        return fwd

    def get_parameters(self) -> List[ParameterTensor]:
        params = []
        for layer in self.layers:
            params.extend(layer.get_parameters())
        return params
