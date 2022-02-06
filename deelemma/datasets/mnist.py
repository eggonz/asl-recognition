from enum import Enum

import numpy as np
import scipy.io
import tensorflow as tf

from .base_dataset import ImageDataset
from ..utils.generic import one_hot


class MnistDataset(Enum):
    # https://www.tensorflow.org/api_docs/python/tf/keras/datasets/mnist
    TF_KERAS = 'TF_KERAS'
    # https://github.com/aurelienduarte/emnist
    EMNIST_MNIST = 'EMNIST_MNIST'
    EMNIST_DIGITS = 'EMNIST_DIGITS'
    EMNIST_LETTERS = 'EMNIST_LETTERS'


class _MnistLoader:
    @staticmethod
    def load_dataset_keras() -> list:
        """Downloads MNIST from tf.keras

        :return: data
        """
        mnist = tf.keras.datasets.mnist
        (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
        return [train_images, train_labels, test_images, test_labels]

    @staticmethod
    def load_dataset_emnist(dataset_name: str, dir_path: str) -> list:
        """Loads emnist dataset specified from a directory

        :param dataset_name: chosen EMNIST dataset name
        :param dir_path: path/to/file/
        :return: data
        """
        emnist_dataset = dataset_name.split('_')[-1].lower()
        file_path = f'{dir_path}emnist-{emnist_dataset}.mat'

        loaded_dataset = scipy.io.loadmat(file_path)['dataset']

        train_images = loaded_dataset['train'][0, 0]['images'][0, 0]
        train_labels = loaded_dataset['train'][0, 0]['labels'][0, 0]
        test_images = loaded_dataset['test'][0, 0]['images'][0, 0]
        test_labels = loaded_dataset['test'][0, 0]['labels'][0, 0]

        return [train_images, train_labels, test_images, test_labels]


class Mnist(ImageDataset):
    _loaded_data = {}  # load datasets once

    def __init__(self,
                 dataset: MnistDataset = MnistDataset.TF_KERAS,
                 dir_path: str = ''):
        """
        Creates MNIST dataset from specified source.

        :param dataset: source
        """
        self.dataset = dataset
        self.dir_path = dir_path

        # Load datasets just once
        if dataset not in self._loaded_data:
            self._load_dataset()
            self._flatten()
            self._normalize_images()
            self.categories = np.sort(np.unique(self.get_train_labels())).tolist()
            self._one_hot_labels()

        self.categories = list(np.sort(np.unique(self.get_train_labels())))

    def get_data(self) -> tuple:
        return self._loaded_data[self.dataset]

    def get_train_images(self) -> np.ndarray:
        return self.get_data()[0]

    def get_train_labels(self) -> np.ndarray:
        return self.get_data()[1]

    def get_test_images(self) -> np.ndarray:
        return self.get_data()[2]

    def get_test_labels(self) -> np.ndarray:
        return self.get_data()[3]

    def _set_train_images(self, data) -> None:
        self._loaded_data[self.dataset][0] = data

    def _set_train_labels(self, data) -> None:
        self._loaded_data[self.dataset][1] = data

    def _set_test_images(self, data) -> None:
        self._loaded_data[self.dataset][2] = data

    def _set_test_labels(self, data) -> None:
        self._loaded_data[self.dataset][3] = data

    def _load_dataset(self) -> None:
        """Loads the dataset specified in the constructor.
        :return: None
        """
        data = None
        if self.dataset == MnistDataset.TF_KERAS:
            data = _MnistLoader.load_dataset_keras()
        elif self.dataset.name.startswith('EMNIST'):
            data = _MnistLoader.load_dataset_emnist(self.dataset.name, self.dir_path)
        self._loaded_data[self.dataset] = data

    def _flatten(self) -> None:
        """Flattens data to convention format with shape
        (samples, flattened_length) for data and (samples, 1) for labels
        :return: None
        """
        shape = self.get_train_images().shape
        self._set_train_images(self.get_train_images().reshape((shape[0], np.prod(shape[1:]))))
        self._set_train_labels(self.get_train_labels().reshape((shape[0], 1)))

        shape = self.get_test_images().shape
        self._set_test_images(self.get_test_images().reshape((shape[0], np.prod(shape[1:]))))
        self._set_test_labels(self.get_test_labels().reshape((shape[0], 1)))

    def _normalize_images(self) -> None:
        """Normalizes each pixel to range [0,1)
        :return: None
        """
        if np.max(self.get_train_images()) < 1:
            return
        self._set_train_images(self.get_train_images() / 255.0)
        self._set_test_images(self.get_test_images() / 255.0)

    def _one_hot_labels(self) -> None:
        """Converts integer labels to one-hot format
        :return: None
        """
        cats = len(self.categories)
        self._set_train_labels(one_hot(self.get_train_labels(), cats))
        self._set_train_labels(one_hot(self.get_test_labels(), cats))

    def __repr__(self):
        return f'{self.__class__.__name__}({self.dataset.name})'


if __name__ == '__main__':
    ds1 = Mnist(MnistDataset.TF_KERAS)
    print(ds1)
    path = '../../test/data/emnist/'
    ds2 = Mnist(MnistDataset.EMNIST_DIGITS, path)
    print(ds2)
    ds3 = Mnist(MnistDataset.EMNIST_MNIST, path)
    print(ds3)
    ds4 = Mnist(MnistDataset.EMNIST_DIGITS, path)
    print(ds4)
