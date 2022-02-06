import random

import PIL.Image
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

from deelemma.datasets.mnist import MnistDataset, Mnist
from deelemma.utils.generic import easy_conv


def save_asl():
    asl_train = 'data/asl_alphabet_db/asl_alphabet_train/asl_alphabet_train/'
    # asl_test = 'data/asl_alphabet_db/asl_alphabet_test/asl_alphabet_test/'

    fig, axs = plt.subplots(3, 3)
    fig.tight_layout()
    for i in range(3):
        for j in range(3):
            let = ['a', 's', 'l'][j]
            n = random.randint(0, 3000)
            path = asl_train + f'{let.upper()}/{let.upper()}{n}.jpg'
            axs[i, j].imshow(mpimg.imread(path))

    plt.savefig('out/misc/asl_kaggle.svg', bbox_inches='tight')
    plt.clf()


def save_mnist():
    mnist = Mnist(MnistDataset.TF_KERAS)

    fig, axs = plt.subplots(1, 5)
    fig.tight_layout()
    plt.subplots_adjust(left=0.1,
                        bottom=0.1,
                        right=0.9,
                        top=0.9,
                        wspace=0.4,
                        hspace=0)
    for j in range(5):
        array = mnist.get_train_images()[j].reshape((28, 28))
        fign = axs[j].imshow(array)
        axs[j].set_axis_off()
        fign.set_cmap('binary')

    plt.savefig('out/misc/mnist.svg', bbox_inches='tight')
    plt.clf()


def save_dog4():
    path = 'data/misc/dog.jpg'
    image = PIL.Image.open(path).convert('RGBA')
    array = np.array(image)  # noqa

    fig, ((axs0, axs1), (axs2, axs3)) = plt.subplots(2, 2)
    fig.tight_layout()

    axs0.imshow(array)
    axs1.imshow(array.mean(axis=2, keepdims=True))
    fig2 = axs2.imshow(array.mean(axis=2, keepdims=True))
    fig2.set_cmap('gray')
    fig3 = axs3.imshow(array.mean(axis=2, keepdims=True))
    fig3.set_cmap('binary')

    plt.savefig('out/misc/dog_0.svg', bbox_inches='tight')
    plt.clf()


def save_conv():
    path = 'data/misc/dog.jpg'
    image = PIL.Image.open(path).convert('RGBA')
    _input = np.array(image)  # noqa

    fig, axs = plt.subplots(2, 2)
    plt.subplots_adjust(left=0.1,
                        bottom=0.06,
                        right=0.9,
                        top=0.94,
                        wspace=0.3,
                        hspace=0.3)

    axs[0, 0].title.set_text('Original')
    axs[0, 0].imshow(_input)

    kernel = [[1, 4, 6, 4, 1],
              [4, 16, 24, 16, 4],
              [6, 24, 36, 24, 6],
              [4, 16, 24, 16, 4],
              [1, 4, 6, 4, 1]]
    _output = easy_conv(_input, kernel)
    axs[0, 1].title.set_text('Gaussian Blur')
    fign = axs[0, 1].imshow(_output)
    fign.set_cmap('gray')

    kernel = [[1, 0, -1],
              [1, 0, -1],
              [1, 0, -1]]
    _output = easy_conv(_input, kernel)
    axs[1, 0].title.set_text('Vertical edges')
    fign = axs[1, 0].imshow(_output)
    fign.set_cmap('gray')

    kernel = [[1, 1, 1],
              [0, 0, 0],
              [-1, -1, -1]]
    _output = easy_conv(_input, kernel)
    axs[1, 1].title.set_text('Horizontal edges')
    fign = axs[1, 1].imshow(_output)
    fign.set_cmap('gray')

    plt.savefig('out/misc/dog_conv.svg', bbox_inches='tight')
    plt.clf()


def save_fake_pooling():
    path = 'data/misc/dog.jpg'
    image = PIL.Image.open(path).convert('L')

    fig, axs = plt.subplots(2, 3)
    axs = axs.reshape(6)
    fig.tight_layout()
    plt.subplots_adjust(left=0.1,
                        bottom=0.1,
                        right=0.9,
                        top=0.9,
                        wspace=0.1,
                        hspace=0)

    def pool_n(n):
        size = 400//(2**n)
        axs[n].title.set_text(f'{size}x{size}')
        axs[n].set_axis_off()
        fign = axs[n].imshow(np.array(image.resize((size, size))))  # noqa
        fign.set_cmap('gray')

    for i in range(6):
        pool_n(i)

    plt.savefig('out/misc/dog_pool.svg', bbox_inches='tight')
    plt.clf()


if __name__ == '__main__':
    pass
    # save_mnist()
    # save_asl()
    # save_conv()
    # save_fake_pooling()
