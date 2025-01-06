import pandas as pd

DATA_PATH = 'dados/dados.csv'

def load_data(path=DATA_PATH):
  df = pd.read_csv(path)
  df['out_sim'] = df['out_sim'].astype(float)
  # df['in_sim'] = df['in_sim'].astype(float)
  df['in_sim'] = df['in_sim'].astype('category')
  df['model'] = df['model'].astype('category')
  df['area'] = df['area'].astype('category')

  return df
