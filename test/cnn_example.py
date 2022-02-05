import activation
import losses
import optimizers
from datasets.asl_alphabet import AslAlphabet
from hyperp import TrainingHyperparameters
from layers import DenseLayer, ActivationLayer, Conv2dLayer, FlattenLayer, MaxPool2dLayer, DropoutLayer
from metrics import Metric
from models import Sequential
from utils.generic import shuffle_data, get_timestamp
from utils.plot import save_train_info

dataset = AslAlphabet('data/asl_alphabet_dataset/', flatten=False, one_hot=True)
print("Successfully loaded:", dataset)

cnn = Sequential([
    Conv2dLayer((200, 200, 3), 32, (3, 3)),  # output = (198, 198, 32)
    ActivationLayer((198, 198, 32), activation.Relu()),
    MaxPool2dLayer((198, 198, 32), (2, 2)),  # output = (99, 99, 32)

    Conv2dLayer((99, 99, 32), 32, (3, 3)),  # output = (97, 97, 32)
    ActivationLayer((97, 97, 32), activation.Relu()),
    MaxPool2dLayer((97, 97, 32), (2, 2)),  # output = (48, 48, 32)

    Conv2dLayer((48, 48, 32), 64, (3, 3)),  # output = (46, 46, 64)
    DropoutLayer((46, 46, 64), 0.3),
    ActivationLayer((46, 46, 64), activation.Relu()),
    MaxPool2dLayer((46, 46, 64), (2, 2)),  # output = (23, 23, 64)

    Conv2dLayer((23, 23, 64), 64, (3, 3)),  # output = (21, 21, 64)
    DropoutLayer((21, 21, 64), 0.3),
    ActivationLayer((21, 21, 64), activation.Relu()),
    MaxPool2dLayer((21, 21, 64), (2, 2)),  # output = (10, 10, 64)

    FlattenLayer((10, 10, 64)),  # output = (6400,)
    DenseLayer((6400,), (128,)),
    DropoutLayer((128,), 0.2),
    ActivationLayer((128,), activation.Relu()),
    DenseLayer((128,), (128,)),
    DropoutLayer((128,), 0.2),
    ActivationLayer((128,), activation.Relu()),
    DenseLayer((128,), (29,)),
    ActivationLayer((29,), activation.Softmax()),
], id_name='test-asl')

params = cnn.get_parameters()
cnn.summary()

# opt = optimizers.SGD(params, lr=1e-2)
opt = optimizers.SGD(params, lr=1e-2, alpha=.7)
# opt = optimizers.RMSProp(params, lr=1e-4, beta=.7)
# opt = optimizers.AdaGrad(params, lr=1e-3)
loss = losses.MSE()
metr = [Metric.LOSS, Metric.CATEGORICAL_ACCURACY]

cnn.compile(optimizer=opt, loss=loss, metrics=metr)

hyperp = TrainingHyperparameters(epochs=30,
                                 batch_size=16,
                                 validation_split=0.85,
                                 # additional info to save
                                 optimizer=opt,
                                 loss=loss)

fold_size = 1024
data_shuffled, label_shuffled = shuffle_data(dataset.get_train_images(), dataset.get_train_labels())

data_fold = data_shuffled[0:fold_size]
label_fold = label_shuffled[0:fold_size]
data_fold_test = data_shuffled[fold_size:2 * fold_size]
label_fold_test = label_shuffled[fold_size:2 * fold_size]

print('pre-train')
cnn.evaluate(data_fold_test, label_fold_test)

history = cnn.fit(data_fold, label_fold, hyperp)
train_id = get_timestamp()
save_train_info(cnn.id, train_id, hyperp, history, show_plot=True)

print('post-train')
cnn.evaluate(data_fold_test, label_fold_test)
