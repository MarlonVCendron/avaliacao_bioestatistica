from load_data import load_data
from labels import labels
import seaborn as sns
import matplotlib.pyplot as plt


def catplot(df):
  sns.set(style="whitegrid")
  g = sns.catplot(
      data=df, kind="bar",
      x="in_sim", y="out_sim", hue="model",
      col="area", ci="sd", palette="muted",
      height=5, aspect=1.2
  )
  g.set_axis_labels(labels['in_sim'], labels['out_sim'])
  g.set_titles(f"{labels['area']}: {{col_name}}")
  g._legend.set_title(labels['model'])

  sns.despine(left=True)
  # plt.show()
  plt.savefig(f'figures/catplot_filter.png')
  plt.close()
  
# Just a single box showing the overall average and outliers of out_sim 
def boxplot(df):
  plt.figure(figsize=(8, 6))
  sns.boxplot(y=df['out_sim'])
  plt.ylabel(labels['out_sim'])
  # plt.show()
  # plt.savefig(f'figures/boxplot.png')
  plt.savefig(f'figures/boxplot_filter.png')
  plt.close()


if __name__ == '__main__':
  df = load_data(True)

  # catplot(df)
  boxplot(df)
