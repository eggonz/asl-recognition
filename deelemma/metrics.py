from enum import Enum
from typing import Optional

import numpy as np
import seaborn as sns

from losses import Loss

sns.set_theme()


class Metric(Enum):
    LOSS = 'LOSS'
    CATEGORICAL_ACCURACY = 'ACC'

    def calc(self, prediction: np.ndarray, target: np.ndarray, loss: Optional[Loss]) -> float:
        """Calculates metric
        :param prediction: predicted output value
        :param target: target output value
        :param loss: Additional parameter - loss function
        :return:
        """

        if self == Metric.LOSS:
            return loss.f(prediction, target)

        if self == Metric.CATEGORICAL_ACCURACY:
            # one-hot encoded data
            correct = (np.argmax(prediction, axis=1) == np.argmax(target, axis=1))
            return np.sum(correct) / correct.size
