from scipy.stats import skew, kurtosis
from scipy.stats import norm
import seaborn as sns
from load_data import load_data
import matplotlib.pyplot as plt
import numpy as np

def skew_kurt(df):
  n = len(df['out_sim'])

  assimetria = skew(df['out_sim'])
  curtose = kurtosis(df['out_sim'], fisher=True)  # Fisher=True retorna a curtose padronizada (g2)

  z_assim = assimetria / (6 / n) ** 0.5
  z_curt = curtose / (24 / n) ** 0.5

  p_assim = 2 * (1 - norm.cdf(abs(z_assim)))  
  p_curt = 2 * (1 - norm.cdf(abs(z_curt)))

  print(f"Assimetria: {assimetria}, Z-Score: {z_assim}, p: {p_assim}")
  print(f"Curtose: {curtose}, Z-Score: {z_curt}, p: {p_curt}")

if __name__ == '__main__':
  df = load_data()
  skew_kurt(df)

