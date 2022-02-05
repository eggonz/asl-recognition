import json
import os

import PIL.Image
import numpy as np

from datasets.base_dataset import ImageDataset
from utils.generic import one_hot, one_hot_decode


class AslAlphabet(ImageDataset):
    # load dataset once
    _loaded_data = None
    _label_map = {}

    def __init__(self, path: str, flatten: bool = False, one_hot: bool = True):
        """Creates ASL Alphabet dataset from specified directory
        :param path: path/to/ where 'asl_alphabet_data.npy' and 'asl_alphabet_label_map.json' files are
        """
        self._path = path
        self._is_flatten = flatten
        self._is_one_hot = one_hot

        if self._loaded_data is None:
            self._load_npy_dataset(path)
        self._cats = len(self._label_map)
        self._train_samples_per_cat = int(self._loaded_data[0].shape[0] / self._cats)
        self._test_samples_per_cat = int(self._loaded_data[2].shape[0] / self._cats)

    def get_data(self) -> tuple:
        return self._loaded_data

    def get_train_images(self) -> np.ndarray:
        train_images = self._loaded_data[0]
        if self._is_flatten:
            return train_images.reshape(train_images.shape[0], -1)
        return train_images

    def get_train_labels(self) -> np.ndarray:
        train_labels = self._loaded_data[1]
        if self._is_one_hot:
            return one_hot(train_labels, self._cats)
        return train_labels

    def get_test_images(self) -> np.ndarray:
        test_images = self._loaded_data[2]
        if self._is_flatten:
            return test_images.reshape(test_images.shape[0], -1)
        return test_images

    def get_test_labels(self) -> np.ndarray:
        test_labels = self._loaded_data[3]
        if self._is_one_hot:
            return one_hot(test_labels, self._cats)
        return test_labels

    def _load_npy_dataset(self, path: str) -> None:
        """Loads the data from the npy and json file"""
        if not os.path.exists(path):
            raise FileNotFoundError
        if 'asl_alphabet_label_map.json' not in os.listdir(path) or 'asl_alphabet_data.npy' not in os.listdir(path):
            raise FileNotFoundError

        with open(f'{path}asl_alphabet_label_map.json', 'r') as f:
            self._label_map = json.load(f, object_hook=lambda x: {int(k): v for k, v in x.items()})

        with open(f'{path}asl_alphabet_data.npy', 'rb') as f:
            train_images = np.load(f)  # noqa
            train_labels = np.load(f)  # noqa
            test_images = np.load(f)  # noqa
            test_labels = np.load(f)  # noqa
        self._loaded_data = (train_images, train_labels, test_images, test_labels)

    def translate_label(self, labels: np.ndarray) -> str:
        """Translates label array to letter representation
        :param labels: array with shape (N,C) or (C) for one-hot encoded data, and (N,1) or (1) if not.
        :return: letter label(s)
        """
        single = False
        if len(labels.shape) == 1:
            single = True
            labels = labels[np.newaxis, :]

        decoded = one_hot_decode(labels) if self._is_one_hot else labels
        letters = np.vectorize(self._label_map.get)(decoded)

        if single:
            letters = letters[0]
        return letters

    @staticmethod
    def build_npy_dataset(src_path: str, dst_path: str, samples_per_class: int = 50) -> None:
        """Reads the images in asl_alphabet_db data folder and builds dataset in npy
        :param src_path: path/to/ where 'asl_alphabet_db/' is
        :param dst_path: path/to/ where 'asl_alphabet_data.npy' and 'asl_alphabet_label_map.json' will be created
        :param samples_per_class: limits the samples that will be loaded from each class
        """
        if not os.path.exists(src_path):
            raise FileNotFoundError
        if 'asl_alphabet_db' not in os.listdir(src_path):
            raise FileNotFoundError
        if not os.path.exists(dst_path):
            os.makedirs(dst_path)

        # Save mapping
        classes = os.listdir(f'{src_path}asl_alphabet_db/asl_alphabet_train/asl_alphabet_train/')
        label_map = {classes.index(c): c for c in classes}
        with open(f'{dst_path}asl_alphabet_label_map.json', 'w') as fp:
            json.dump(label_map, fp)

        # Load images
        def load_classes(path):
            cats = os.listdir(path)
            mapping = {c: cats.index(c) for c in cats}
            data = []
            labels = []
            print('Loading:', end=' ')
            for cat in cats:
                print(cat, end='-')

                files = os.listdir(f'{path}{cat}/')
                samples_to_load = np.random.choice(files, min(samples_per_class, len(files)))
                for image in samples_to_load:
                    img = PIL.Image.open(f'{path}{cat}/{image}').convert('RGB')
                    data.append(np.array(img) / 255.0)  # noqa
                    labels.append(mapping[cat])

            print('>DONE')
            data = np.array(data)
            labels = np.array(labels)[:, np.newaxis]
            return data, labels

        train_images, train_labels = load_classes(f'{src_path}asl_alphabet_db/asl_alphabet_train/asl_alphabet_train/')
        test_images, test_labels = load_classes(f'{src_path}asl_alphabet_db/asl_alphabet_test/asl_alphabet_test/')

        with open(f'{dst_path}asl_alphabet_data.npy', 'wb') as f:
            np.save(f, train_images)  # noqa
            np.save(f, train_labels)  # noqa
            np.save(f, test_images)  # noqa
            np.save(f, test_labels)  # noqa

    def __repr__(self):
        return f'{self.__class__.__name__}' \
               f'[{self._cats}x{self._train_samples_per_cat}|{self._test_samples_per_cat}]' \
               f'(path=\'{self._path}\', flatten={str(self._is_flatten)}, one_hot={str(self._is_one_hot)})'


if __name__ == '__main__':
    # 29 cats
    # train: 3000/cat (tot=87000)
    # test: 1/cat
    # RGB img shape: (200, 200, 3)
    # alpha=255 in all, RGBA not necessary

    db_path = '../test/data/'
    ds_path = '../test/data/asl_alphabet_dataset/'
    AslAlphabet.build_npy_dataset(db_path, ds_path, 100)
    print('Dataset saved')

    dataset = AslAlphabet(ds_path, flatten=False, one_hot=True)
    print(dataset)
    print(dataset.get_train_images().shape)
    print(dataset.get_train_labels().shape)
    print(dataset.get_test_images().shape)
    print(dataset.get_test_labels().shape)

    dataset = AslAlphabet(ds_path, flatten=False, one_hot=False)
    print(dataset)
    print(dataset.get_train_images().shape)
    print(dataset.get_train_labels().shape)
    print(dataset.get_test_images().shape)
    print(dataset.get_test_labels().shape)

    dataset = AslAlphabet(ds_path, flatten=True, one_hot=True)
    print(dataset)
    print(dataset.get_train_images().shape)
    print(dataset.get_train_labels().shape)
    print(dataset.get_test_images().shape)
    print(dataset.get_test_labels().shape)
