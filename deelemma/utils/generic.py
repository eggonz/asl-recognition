import datetime
import importlib
from enum import Enum
from typing import Tuple, Any

import numpy as np
import scipy.signal
import tensorflow as tf
from PIL import Image, ImageOps

from utils.exceptions import InvalidShapeException


def load_image(path: str) -> np.ndarray:
    image = Image.open(path)
    inverted = ImageOps.invert(image)
    return np.array(inverted.getdata())


def get_timestamp() -> str:
    # local time, not utc
    return datetime.datetime.now().isoformat().replace('-', '').replace(':', '').replace('.', '')


def shuffle_data(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Shuffles dataset samples (data and labels)
    :param x: training sample data
    :param y: training examples labels
    :return: x_shuffled, y_shuffled
    """
    c = list(zip(x, y))
    np.random.shuffle(c)
    x_aux, y_aux = zip(*c)
    x_shuffled, y_shuffled = np.array(x_aux), np.array(y_aux)
    return x_shuffled, y_shuffled


def split_data(x: np.ndarray, y: np.ndarray, split: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Shuffles and splits datasets samples (axis=0 of input arrays)
    :param x: input of the training examples
    :param y: output of the training examples
    :param split: training-validation split. 'split' for train and '1-split' for validation
    :return: x_train, x_valid, y_train, y_valid
    """
    # Shuffle
    x_shuffled, y_shuffled = shuffle_data(x, y)

    # Split
    n_samples = x.shape[0]
    split_index = int(n_samples * split)
    return x_shuffled[:split_index], x_shuffled[split_index:], y_shuffled[:split_index], y_shuffled[split_index:]


def get_class_instance(module_name: str, class_name: str) -> Any:
    module = importlib.import_module(module_name)
    class_ = getattr(module, class_name)
    instance = class_()
    return instance


class ConvPadding(Enum):
    VALID = 'VALID'
    SAME = 'SAME'
    FULL = 'FULL'


def conv2d(input_tensor: np.ndarray, kernel_tensor: np.ndarray,
           padding: ConvPadding = ConvPadding.VALID) -> np.ndarray:
    """Conv2D operation
    - VALID padding: (N, H, W, C) * (KH, KW, C, F) -> (N, H-KH+1, W-KW+1, F)
    - SAME padding: (N, H, W, C) * (KH, KW, C, F) -> (N, H, W, F)
    - FULL padding: (N, H, W, C) * (KH, KW, C, F) -> (N, H+KH-1, W+KW-1, F)
    :param input_tensor:
    :param kernel_tensor:
    :param padding: padding algorithm
    :return:
    """
    if len(input_tensor.shape) != 4 or len(kernel_tensor.shape) != 4:
        raise InvalidShapeException('Invalid shapes for convolution operation')
    if input_tensor.shape[3] != kernel_tensor.shape[2]:
        raise InvalidShapeException('Invalid shapes for convolution operation')

    if padding == ConvPadding.FULL:
        kh, kw, _, _ = kernel_tensor.shape
        pad = [[0, 0], [kh - 1, kh - 1], [kw - 1, kw - 1], [0, 0]]
    else:
        pad = padding.value

    # https://www.tensorflow.org/api_docs/python/tf/nn/conv2d
    output_tensor = tf.nn.conv2d(tf.constant(input_tensor), tf.constant(kernel_tensor),
                                 strides=1, padding=pad).numpy()
    return output_tensor


def maxpool2d(input_tensor: np.ndarray, kernel_shape: Tuple[int, int]) -> np.ndarray:
    """MaxPool2D operation
    (N, H, W, C) * (KH, KW) -> (N, H//KH, W//KW, F)
    :param input_tensor:
    :param kernel_shape: shape of the pooling kernel
    :return:
    """
    if len(input_tensor.shape) != 4 or len(kernel_shape) != 2:
        raise InvalidShapeException('Invalid shapes for pooling operation')

    # https://www.tensorflow.org/api_docs/python/tf/nn/max_pool2d
    output_tensor = tf.nn.max_pool2d(tf.constant(input_tensor), kernel_shape,
                                     strides=kernel_shape, padding='VALID').numpy()
    return output_tensor


def rot180(tensor: np.ndarray) -> np.ndarray:
    """Rotates a tensor 180 in the first 2 axes"""
    return np.rot90(tensor, 2, (0, 1))


def one_hot(array: np.ndarray, cats: int) -> np.ndarray:
    """Returns the array categories one-hot encoded
    :param array: input array with shape (N,1) and values<_cats
    :param cats: categories, number of classes
    :return: one-hot encoded array
    """
    return np.eye(cats)[array[:, 0]]


def one_hot_decode(encoded: np.ndarray) -> np.ndarray:
    """Returns the array categories one-hot encoded
    :param encoded: one-hot encoded array with shape (N,C)
    :return: decoded array with shape (N,1)
    """
    return np.argmax(encoded, axis=1, keepdims=True)


def easy_conv(input_array, kernel_list):
    """Performs 2-D convolution between rank-2 arrays. If the input array contains channels (RGB) in a
    third dimension, the average value is calculated, and converted to a single dimension (grayscale)"""
    tensor = input_array.mean(axis=2)
    kernel = np.array(kernel_list)
    return scipy.signal.convolve(tensor, kernel, mode='valid')
