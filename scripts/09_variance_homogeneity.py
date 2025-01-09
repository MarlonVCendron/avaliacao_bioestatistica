from load_data import load_data
from scipy.stats import levene

if __name__ == '__main__':
  df = load_data()

  groups = [group['out_sim'].values for name, group in df.groupby(['area', 'model', 'in_sim'])]

  stat, p_value = levene(*groups)

  print(f"Levene: {stat}")
  print(f"P: {p_value}")

  if p_value < 0.05:
    print("Variâncias diferem")
  else:
    print("Variâncias homogêneas")
