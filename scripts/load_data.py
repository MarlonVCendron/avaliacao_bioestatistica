import pandas as pd

DATA_PATH = 'dados/dados.csv'

def load_data(filter_outliers=True, path=DATA_PATH):
  df = pd.read_csv(path)
  df['out_sim'] = df['out_sim'].astype(float)
  # df['in_sim'] = df['in_sim'].astype(float)
  df['in_sim'] = df['in_sim'].astype('category')
  df['model'] = df['model'].astype('category')
  df['area'] = df['area'].astype('category')

  # Remove in_sim 0 e 1
  if filter_outliers:
    df = df.loc[~df['in_sim'].isin([0, 1])]
    df['in_sim'] = df['in_sim'].cat.remove_unused_categories()


  return df
