import numpy as np
import pandas as pd


def calculate_alpha_beta(mean, std_dev):
  variance = std_dev**2
  alpha = ((1 - mean) / variance - 1 / mean) * mean**2
  beta = alpha * (1 / mean - 1)
  return alpha, beta


def export_data(data, output_file):
  melted_df = data.melt(var_name="variable", value_name="out_sim")

  melted_df[['model', 'area', 'in_sim']] = melted_df['variable'].str.extract(r'(\w+)_(\w+)_sim(\d+)')
  melted_df['in_sim'] = melted_df['in_sim'].astype(float) / 100

  melted_df = melted_df.drop(columns=['variable'])
  melted_df = melted_df[['out_sim', 'in_sim', 'model', 'area']]

  print(melted_df)
  melted_df.to_csv(output_file, index=False)
  print(f"Dados exportados para: {output_file}")


def generate_random_data(num_samples, mean, std_dev):
  return np.random.normal(loc=mean, scale=std_dev, size=num_samples)


def generate_beta_random_data(num_samples, mean, std_dev):
  alpha, beta = calculate_alpha_beta(mean, std_dev)
  return np.random.beta(a=alpha, b=beta, size=num_samples)


def generate_data(num_samples, column_params, output_file="multi_column_data.csv"):
  data = {}
  for column_name, mean, std_dev in column_params:
    data[column_name] = generate_beta_random_data(num_samples, mean, std_dev)

  df = pd.DataFrame(data)
  export_data(df, output_file)


num_samples = 10
output_file = "output_similarity.csv"
# column_name, mean, std_dev
column_params = [
    # c dg
    ("c_dg_sim00", 0.001, 0.0001),
    ("c_dg_sim10", 0.15, 0.04),
    ("c_dg_sim20", 0.15, 0.05),
    ("c_dg_sim30", 0.20, 0.04),
    ("c_dg_sim40", 0.25, 0.04),
    ("c_dg_sim50", 0.28, 0.05),
    ("c_dg_sim60", 0.30, 0.04),
    ("c_dg_sim70", 0.60, 0.04),
    ("c_dg_sim80", 0.70, 0.04),
    ("c_dg_sim90", 0.78, 0.04),
    ("c_dg_sim100", 0.99, 0.001),
    # c ca3
    ("c_ca3_sim00", 0.001, 0.0001),
    ("c_ca3_sim10", 0.15, 0.04),
    ("c_ca3_sim20", 0.15, 0.04),
    ("c_ca3_sim30", 0.25, 0.04),
    ("c_ca3_sim40", 0.30, 0.04),
    ("c_ca3_sim50", 0.30, 0.05),
    ("c_ca3_sim60", 0.45, 0.05),
    ("c_ca3_sim70", 0.60, 0.04),
    ("c_ca3_sim80", 0.80, 0.04),
    ("c_ca3_sim90", 0.95, 0.05),
    ("c_ca3_sim100", 0.98, 0.001),

    # ng dg
    ("ng_dg_sim00", 0.001, 0.0001),
    ("ng_dg_sim10", 0.05, 0.05),
    ("ng_dg_sim20", 0.15, 0.04),
    ("ng_dg_sim30", 0.10, 0.04),
    ("ng_dg_sim40", 0.15, 0.04),
    ("ng_dg_sim50", 0.18, 0.04),
    ("ng_dg_sim60", 0.18, 0.04),
    ("ng_dg_sim70", 0.13, 0.04),
    ("ng_dg_sim80", 0.16, 0.04),
    ("ng_dg_sim90", 0.18, 0.04),
    ("ng_dg_sim100", 0.99, 0.001),
    # ng ca3
    ("ng_ca3_sim00", 0.001, 0.0001),
    ("ng_ca3_sim10", 0.05, 0.04),
    ("ng_ca3_sim20", 0.11, 0.04),
    ("ng_ca3_sim30", 0.19, 0.04),
    ("ng_ca3_sim40", 0.08, 0.04),
    ("ng_ca3_sim50", 0.17, 0.05),
    ("ng_ca3_sim60", 0.19, 0.04),
    ("ng_ca3_sim70", 0.22, 0.04),
    ("ng_ca3_sim80", 0.23, 0.05),
    ("ng_ca3_sim90", 0.15, 0.04),
    ("ng_ca3_sim100", 0.98, 0.01),

    # ngt dg
    ("ngt_dg_sim00", 0.001, 0.0001),
    ("ngt_dg_sim10", 0.04, 0.04),
    ("ngt_dg_sim20", 0.04, 0.04),
    ("ngt_dg_sim30", 0.12, 0.05),
    ("ngt_dg_sim40", 0.10, 0.05),
    ("ngt_dg_sim50", 0.12, 0.05),
    ("ngt_dg_sim60", 0.08, 0.05),
    ("ngt_dg_sim70", 0.11, 0.05),
    ("ngt_dg_sim80", 0.11, 0.05),
    ("ngt_dg_sim90", 0.13, 0.05),
    ("ngt_dg_sim100", 0.99, 0.01),
    # ngt ca3
    ("ngt_ca3_sim00", 0.001, 0.0001),
    ("ngt_ca3_sim10", 0.05, 0.05),
    ("ngt_ca3_sim20", 0.06, 0.05),
    ("ngt_ca3_sim30", 0.09, 0.05),
    ("ngt_ca3_sim40", 0.11, 0.05),
    ("ngt_ca3_sim50", 0.15, 0.05),
    ("ngt_ca3_sim60", 0.14, 0.05),
    ("ngt_ca3_sim70", 0.15, 0.05),
    ("ngt_ca3_sim80", 0.13, 0.05),
    ("ngt_ca3_sim90", 0.08, 0.05),
    ("ngt_ca3_sim100", 0.98, 0.01),
]

generate_data(num_samples, column_params, output_file)
