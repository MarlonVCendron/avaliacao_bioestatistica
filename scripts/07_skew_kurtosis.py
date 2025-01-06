from scipy.stats import skew, kurtosis
from load_data import load_data

if __name__ == '__main__':
  df = load_data()

  assimetria = skew(df['out_sim'])
  curtose = kurtosis(df['out_sim'], fisher=True)  # Fisher=True retorna a curtose padronizada (g2)
  print(f"Assimetria: {assimetria}")
  print(f"Curtose: {curtose}")
