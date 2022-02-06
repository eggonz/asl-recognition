import os

import h5py

from ..layers import DenseLayer, ActivationLayer, DropoutLayer
from ..models import Sequential
from ..parameters import ParameterInitializer
from .generic import get_timestamp, get_class_instance


def save_model(nn: Sequential) -> str:
    """Saves model in "models" directory
    :param nn: neural network model
    :return: 'path/to/model.h5'
    """
    path = f'out/models/{nn.id}'
    if not os.path.exists(path):
        os.makedirs(path)
    file_path = os.path.join(path, f'model_{get_timestamp()}.h5')

    with h5py.File(file_path, 'w') as h5:

        h5.attrs['id'] = nn.id

        for i, layer in enumerate(nn.layers):
            layer_class = layer.__class__.__name__

            group = h5.create_group(f'layer{i}')
            group.attrs['class'] = layer_class
            group.attrs['dims'] = (layer.input_shape[1:], layer.output_shape[1:])

            if layer_class == 'DenseLayer':
                layer: DenseLayer
                group.create_dataset('weights', layer.weights.value.shape, 'f', layer.weights.value)
                group.create_dataset('bias', layer.bias.value.shape, 'f', layer.bias.value)
            elif layer_class == 'ActivationLayer':
                layer: ActivationLayer
                group.attrs['act'] = layer.act.__class__.__name__
            elif layer_class == 'DropoutLayer':
                layer: DropoutLayer
                group.attrs['dropout_rate'] = layer.dropout_rate

    return file_path


def load_model(file_path: str) -> Sequential:
    """Loads model from given file
    :param file_path: h5 file with model
    :return: neural network model
    """
    model = []

    with h5py.File(file_path, 'r') as h5:

        id_name = h5.attrs['id']

        for group in h5.values():
            layer_class = group.attrs['class']
            input_shape, output_shape = group.attrs['dims']

            if layer_class == 'DenseLayer':
                layer = DenseLayer(input_shape, output_shape, ParameterInitializer.ZEROS)
                group['weights'].read_direct(layer.weights.value)
                group['bias'].read_direct(layer.bias.value)
            elif layer_class == 'ActivationLayer':
                act = group.attrs['act']
                act_ins = get_class_instance('activation', act)
                layer = ActivationLayer(input_shape, act_ins)
            elif layer_class == 'DropoutLayer':
                dropout_rate = float(group.attrs['dropout_rate'])
                layer = DropoutLayer(input_shape, dropout_rate)

            model.append(layer)

    return Sequential(model, id_name=id_name)
