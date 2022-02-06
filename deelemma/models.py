from abc import abstractmethod
from typing import List, Dict, Optional

import numpy as np

from .base_layer import LayerGroup, Layer
from .hyperp import TrainingHyperparameters
from .layers import ActivationLayer, DropoutLayer, BatchNormLayer
from .losses import Loss
from .metrics import Metric
from .optimizers import Optimizer
from .parameters import ParameterTensor
from .utils.exceptions import InvalidShapeException, NotConfiguredException
from .utils.generic import split_data
from .utils.plot import print_cols, print_cols_center, SUMMARY_COL_WIDTH


class Model:
    def __init__(self, id_name: str):
        self.id = id_name

        self._optimizer: Optional[Optimizer] = None
        self._loss: Optional[Loss] = None
        self._metrics: Optional[List[Metric]] = None
        self._is_compiled = False

        self._hist = {}

    @abstractmethod
    def summary(self) -> None:
        """Prints model summary"""
        pass

    def compile(self, optimizer: Optimizer, loss: Loss, metrics: List[Metric]) -> None:
        """Configures and prepares model for training
        :param optimizer: optimizer
        :param loss: loss function
        :param metrics: list of desired metrics
        """
        self._optimizer = optimizer
        self._loss = loss
        self._metrics = metrics
        self._is_compiled = True

    def fit(self, x: np.ndarray, y: np.ndarray, hyperp: TrainingHyperparameters,
            verbose: bool = True) -> Dict[str, List[float]]:
        """Fits model to provided data. Needs to be compiled before
        :param x: input data
        :param y: target output data
        :param hyperp: training hyperparameters
        :param verbose: whether it should show training process info
        :return: metrics history
        """
        if not self._is_compiled:
            raise NotConfiguredException('Model not compiled')
        if x.shape[0] != y.shape[0]:
            raise InvalidShapeException('Incompatible training samples in input and output')

        return self._fit(x, y, hyperp, verbose)

    def _fit(self, x: np.ndarray, y: np.ndarray, hyperp: TrainingHyperparameters,
             verbose: bool = True) -> Dict[str, List[float]]:
        # Per-epoch metrics history
        self._hist = {metric.value + data: [] for metric in self._metrics for data in ['_train', '_valid']}

        # Data
        x_train, x_valid, y_train, y_valid = split_data(x, y, hyperp.validation_split)

        for epoch in range(hyperp.epochs):
            if verbose:
                print(f'Epoch: {epoch + 1}/{hyperp.epochs} ', end='')
            self._perform_training_epoch(x_train, x_valid, y_train, y_valid, hyperp, verbose)

        return self._hist

    @abstractmethod
    def _perform_training_epoch(self, x_train: np.ndarray, x_valid: np.ndarray,
                                y_train: np.ndarray, y_valid: np.ndarray, hyperp: TrainingHyperparameters,
                                verbose: bool) -> None:
        """Performs training epoch
        :param x_train: input training data
        :param x_valid: input validation data
        :param y_train: target output training data
        :param y_valid: target output validation data
        :param hyperp: training hyperparameters
        :param verbose: show info about training process
        :return: None
        """
        pass

    def evaluate(self, x: np.ndarray, y: np.ndarray, verbose: bool = True) -> Dict[str, float]:
        """Tests the model on the given data. Needs to be compiled before
        :param x: input data
        :param y: target output data
        :param verbose: show info about testing process
        :return: metrics history
        """
        if not self._is_compiled:
            raise NotConfiguredException('Model not compiled')
        if x.shape[0] != y.shape[0]:
            raise InvalidShapeException('Incompatible training samples in input and output')

        eval_out = self.predict(x)

        hist = {metric.value + '_test': 0 for metric in self._metrics}
        for metric in self._metrics:
            hist[metric.value + '_test'] = metric.calc(eval_out, y, self._loss)
            if verbose:
                print(metric.value + '_test', round(hist[metric.value + '_test'], 4), end=' ')
        if verbose:
            print()
        return hist

    @abstractmethod
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Generates an output for the given data
        :param x: input data
        :return: generated output
        """
        pass

    def get_parameters(self) -> List[ParameterTensor]:
        """Returns all the model parameters
        :return: a list with the parameters tensors of the model
        """
        pass


class Sequential(Model):
    def __init__(self, layers: List[Layer], id_name: str):
        super(Sequential, self).__init__(id_name)
        self._layers = LayerGroup(layers)

    @property
    def layers(self):
        return self._layers.layers

    def _fit(self, x: np.ndarray, y: np.ndarray, hyperp: TrainingHyperparameters,
             verbose: bool = True) -> Dict[str, List[float]]:
        self._layers.on_training_start()
        res = super(Sequential, self)._fit(x, y, hyperp, verbose)
        self._layers.on_training_end()
        return res

    def _perform_training_iteration(self, x: np.ndarray, y: np.ndarray) -> None:
        """Performs training iteration: forward, backward and optimize
        :param x: input data
        :param y: target output data
        :return: None
        """
        self._layers.on_iteration_start()
        out = self._layers.forward(x)
        self._layers.backward(self._loss.df(out, y))
        self._optimizer.step()
        self._layers.on_iteration_end()

    def _perform_training_epoch(self, x_train: np.ndarray, x_valid: np.ndarray, y_train: np.ndarray,
                                y_valid: np.ndarray, hyperp: TrainingHyperparameters, verbose: bool) -> None:
        self._layers.on_epoch_start()
        n_samples = x_train.shape[0]
        if verbose:
            verbose_batch = 0

        self._optimizer.reset()

        batch_num = int(np.ceil(n_samples / hyperp.batch_size))

        # For calculating per-epoch minibatch training loss approximation
        minibatch_loss = {metric.value: [] for metric in self._metrics}

        for batch_index in range(batch_num):
            # Initial and final indices of batch
            bi = batch_index * hyperp.batch_size
            bf = min(bi + hyperp.batch_size, n_samples)

            # Forward and backward pass in each iteration
            self._perform_training_iteration(x_train[bi:bf], y_train[bi:bf])

            # Additional evaluation to save the correct metric value
            eval_out = self._layers.inference(x_train[bi:bf])

            # Compute approximate minibatch metrics
            for metric in self._metrics:
                minibatch_loss[metric.value].append(metric.calc(eval_out, y_train[bi:bf], self._loss))

            if verbose:
                if batch_index / batch_num > verbose_batch:  # noqa
                    verbose_batch += .05
                    print('#', end='')

        eval_out = self._layers.inference(x_valid)
        for metric in self._metrics:
            self._hist[metric.value + '_train'].append(np.mean(minibatch_loss[metric.value]))
            self._hist[metric.value + '_valid'].append(metric.calc(eval_out, y_valid, self._loss))
            if verbose:
                print(' ', end='')
                print(metric.value + '_train', round(self._hist[metric.value + '_train'][-1], 4),
                      metric.value + '_valid', round(self._hist[metric.value + '_valid'][-1], 4), end='')
        if verbose:
            print()
        self._layers.on_epoch_end()

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self._layers.inference(x)

    def get_parameters(self) -> List[ParameterTensor]:
        return self._layers.get_parameters()

    def summary(self) -> None:
        # Header and model info
        print_cols_center('=' * SUMMARY_COL_WIDTH, '=' * SUMMARY_COL_WIDTH, 'MODEL SUMMARY',
                          '=' * SUMMARY_COL_WIDTH, '=' * SUMMARY_COL_WIDTH)
        print_cols('id', self.id)
        for k in ['input_shape', 'output_shape', 'depth']:
            print_cols(k, self._layers.__dict__.get(k, ''))
        print_cols('total params', sum(len(p) for p in self._layers.get_parameters()))
        print()

        # Layers header
        print_cols('LAYER', 'INPUT', 'OUTPUT', 'PARAMS', 'INFO')
        print_cols(*['-' * (SUMMARY_COL_WIDTH // 2)] * 5)

        def print_layer_group(layer_group: LayerGroup) -> None:
            for layer in layer_group.layers:
                if isinstance(layer, LayerGroup):
                    print_layer_group(layer)
                else:
                    print_layer_row(layer)

        def print_layer_row(layer: Layer) -> None:
            excluded = (ActivationLayer, DropoutLayer, BatchNormLayer)
            in_shape = layer.input_shape if type(layer) not in excluded else ''
            out_shape = layer.output_shape if type(layer) not in excluded else ''
            params = sum(len(p) for p in layer.get_parameters())
            info = layer.get_summary_info()
            print_cols(layer.__class__.__name__, in_shape, out_shape, params if params != 0 else '', info)

        print_layer_group(self._layers)
        print_cols_center(*['=' * SUMMARY_COL_WIDTH] * 5)
