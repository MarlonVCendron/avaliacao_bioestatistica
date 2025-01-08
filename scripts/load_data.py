import pandas as pd

humano = 'marlon'

DATA_PATH = 'dados/dados.csv' if humano == 'marlon' else r'C:\Users\pcost\Desktop\biostat\dados.csv'


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

if __name__ == '__main__': 
  df = load_data()
  stats = df.groupby(['area', 'model'])['out_sim'].agg(['mean', 'std', 'min', 'max', 'median', 'count'])
  stats = stats.reset_index()
  output = pd.DataFrame (stats)
  output.to_csv(r'C:\Users\pcost\Desktop\biostat\descritivas.csv', index=False)

