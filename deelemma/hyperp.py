# Structure hyperp: topology, weight_init, activation + dropout_percentage, regularization?
# Training hyperp: learning_rate, epochs, batch_size, validation_split + momentum
# Hyperp tuning methods: Manual search / Grid search / Random search // Bayesian optimization


class TrainingHyperparameters:
    def __init__(self, epochs: int, batch_size: int, validation_split: float, **kwargs):
        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_split = validation_split

        self._kwargs = kwargs

    def as_dict(self) -> dict:
        return {'epochs': self.epochs, 'batch_size': self.batch_size,
                'validation_split': self.validation_split, **self._kwargs}

    def __str__(self) -> str:
        return '\n'.join([f'{k}={v}' for k, v in self.as_dict().items()])


if __name__ == '__main__':
    h = TrainingHyperparameters(1, 3, .2, a=3)
    d1 = h.as_dict()
    print(d1)
    h = TrainingHyperparameters(**d1)
    d2 = h.as_dict()
    print(d2)
    print(d1 == d2)
