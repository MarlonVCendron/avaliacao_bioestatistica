from load_data import load_data
from labels import labels
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np



def hist_all(df):

  data = df['out_sim']
 
  fig, ax = plt.subplots(figsize =(8, 6))
  sns.histplot(data, bins=20, kde=False, ax=ax, edgecolor=None)
 
  ax.set_xlabel(labels['out_sim'])
  ax.set_ylabel('Frequência')
  plt.suptitle(f'Histograma para {labels["out_sim"]} (independente de categorias)')

  fig.tight_layout()

  plt.show()


if __name__ == '__main__':
  df = load_data()
  hist_all(df)
