from load_data import load_data
from labels import labels
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

MAX_COLS = 3


def plot_histogram(group_col, df):
  unique_groups = df[group_col].unique()
  num_groups = len(unique_groups)
  cols = num_groups if num_groups < MAX_COLS else MAX_COLS
  rows = int(np.ceil(num_groups / MAX_COLS))

  fig, axes = plt.subplots(rows, cols, figsize=(3*MAX_COLS, 3*rows))
  axes = axes.flatten()

  for i, group in enumerate(unique_groups):
    ax = axes[i] if num_groups > 1 else axes
    data = df[df[group_col] == group]['out_sim']
    sns.histplot(data, bins=20, kde=False, ax=ax, edgecolor=None)

    ax.set_title(f"{labels[group_col]}: {group}")
    ax.set_xlabel(labels['out_sim'])
    ax.set_ylabel('Frequência')

  fig.suptitle(f'Histogramas de {labels["out_sim"]} por {labels[group_col]}')
  fig.tight_layout()

  plt.show()
  # plt.savefig(f'figures/hist_{group_col}.png')
  # plt.close(fig)


if __name__ == '__main__':
  df = load_data()

  plot_histogram('in_sim', df)
  plot_histogram('model', df)
  plot_histogram('area', df)
