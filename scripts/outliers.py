from scipy.stats import zscore
from scipy.stats import norm
import seaborn as sns
from load_data import load_data
import matplotlib.pyplot as plt

def outliers(df):
    fig, ax = plt.subplots(figsize =(8, 6))
    sns.boxplot(data=df['out_sim'])
    ax.set_ylabel('Similaridade de Saída')
    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
  df = load_data()
  outliers(df)
