from load_data import load_data
from labels import labels
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np

# pd.set_option('display.max_rows', None, 'display.max_columns', None)


def test_shapiro(data):
  stat, p = stats.shapiro(data)
  return p


def test_kolmogorov_smirnov(data):
  stat, p = stats.kstest(data, 'norm')
  return p


if __name__ == '__main__':
  df = load_data()
  # normality_results = df.groupby(['model', 'in_sim', 'area'])['out_sim'].apply(test_kolmogorov_smirnov)
  normality_results = test_kolmogorov_smirnov(df['out_sim'])
  print("Teste de Normalidade:")
  print(normality_results)
  # print(kstest_results)
  # print(f'Total normais: {normality_results.sum()}')
