import os

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from hyperp import TrainingHyperparameters

SUMMARY_COL_WIDTH = 25


def print_cols(*ls) -> None:
    """The arguments passed are printed in separate columns, left-aligned in the column"""
    print("".join([str(s).ljust(SUMMARY_COL_WIDTH) for s in ls]))


def print_cols_right(*ls) -> None:
    """The arguments passed are printed in separate columns, right-aligned in the column"""
    print("".join([str(s).rjust(SUMMARY_COL_WIDTH) for s in ls]))


def print_cols_center(*ls) -> None:
    """The arguments passed are printed in separate columns, centered in the column"""
    print("".join([str(s).center(SUMMARY_COL_WIDTH) for s in ls]))


def save_train_info(model_id: str, train_id: str, hyperp: TrainingHyperparameters, hist: dict,
                    show_plot: bool = True) -> None:
    """Saves training information in dir 'out/models/{model_id}/trainings/{train_id}/'
    The following files are generated:
    - metrics.csv: history of metric values
    - hyperp.txt: summary of the hyperparameter configuration used in training
    - plot.png, plot.pdf, plot.svg: metric evolution plot in several formats
    Additionally, this function can also show the saved plots in a new window.
    """
    path = f'out/models/{model_id}/trainings/{train_id}'
    if not os.path.exists(path):
        os.makedirs(path)

    df = pd.DataFrame.from_dict(hist)
    # Metrics hist
    df.to_csv(os.path.join(path, 'metrics.csv'), index=False)  # noqa

    # Hyperp info
    with open(os.path.join(path, 'hyperp.txt'), 'w') as file:
        file.write(str(hyperp))

    # Evolution plot
    df['x'] = np.arange(1, len(df) + 1)
    df = df.melt('x', var_name='Metric', value_name='vals')
    sns.lineplot(x='x', y='vals', hue='Metric', data=df)
    plt.title(f'{model_id}_{train_id}')
    plt.xlabel('epochs')
    plt.ylabel('metrics')
    plt.savefig(os.path.join(path, f'plot.png'))
    plt.savefig(os.path.join(path, f'plot.pdf'))
    plt.savefig(os.path.join(path, f'plot.svg'))

    if show_plot:
        plt.show()  # FIXME plot blocks program execution
    plt.clf()
