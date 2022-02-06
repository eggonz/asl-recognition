import time

from deelemma import activation
from deelemma import losses
from deelemma import optimizers
from deelemma.datasets.mnist import Mnist, MnistDataset
from deelemma.hyperp import TrainingHyperparameters
from deelemma.layers import DenseLayer, ActivationLayer, DropoutLayer
from deelemma.metrics import Metric
from deelemma.models import Sequential
from deelemma.utils.generic import get_timestamp, shuffle_data
from deelemma.utils.plot import save_train_info

# Test to download dataset
mnist = Mnist(MnistDataset.EMNIST_MNIST, dir_path='data/emnist/')
print("Successfully loaded:", mnist)

# Example
mlp = Sequential([
    DenseLayer((784,), (128,)),
    # DropoutLayer((256,), .1),
    ActivationLayer((128,), activation.Relu()),
    DenseLayer((128,), (128,)),
    DropoutLayer((128,), .1),
    ActivationLayer((128,), activation.Relu()),
    DenseLayer((128,), (10,)),
    ActivationLayer((10,), activation.Softmax()),
], id_name='test-mnist')

mlp.summary()

# opt = optimizers.SGD(mlp.get_parameters(), lr=1e-2)
opt = optimizers.SGD(mlp.get_parameters(), lr=1e-2, alpha=.7)
# opt = optimizers.RMSProp(mlp.get_parameters(), lr=1e-4, beta=.8)
# opt = optimizers.AdaGrad(mlp.get_parameters(), lr=1e-3)
loss = losses.MSE()
metr = [Metric.LOSS, Metric.CATEGORICAL_ACCURACY]

mlp.compile(optimizer=opt, loss=loss, metrics=metr)

hyperp = TrainingHyperparameters(epochs=30,
                                 batch_size=16,
                                 validation_split=0.85,
                                 # additional info to save
                                 optimizer=opt,
                                 loss=loss)

fold_size = 1000
data_shuffled, label_shuffled = shuffle_data(mnist.get_train_images(), mnist.get_train_labels())
data_fold = data_shuffled[0:fold_size]
label_fold = label_shuffled[0:fold_size]
data_fold_test = data_shuffled[fold_size:2 * fold_size]
label_fold_test = label_shuffled[fold_size:2 * fold_size]

print('pre-train')
mlp.evaluate(data_fold_test, label_fold_test)
print(mnist.get_train_labels()[0])
p = mlp.predict(mnist.get_train_images()[0])[0]
print((p == max(p)).astype(float))  # noqa

t1 = time.time()
history = mlp.fit(data_fold, label_fold, hyperp)
t2 = time.time()
print(f'Training time: {t2 - t1}')
train_id = get_timestamp()
save_train_info(mlp.id, train_id, hyperp, history, show_plot=True)

print('post-train')
mlp.evaluate(data_fold_test, label_fold_test)
print(mnist.get_train_labels()[0])
p = mlp.predict(mnist.get_train_images()[0])[0]
print((p == max(p)).astype(float))  # noqa
