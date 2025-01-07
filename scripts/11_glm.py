from load_data import load_data
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf

if __name__ == '__main__':
  df = load_data()
  formula = 'out_sim ~ model + area + model:area'
  glm = smf.glm(formula=formula, data=df, family=sm.families.Gaussian()).fit()
  print("Resultados dos Modelos Lineares Generalizados:")
  print(glm.summary())

  sns.set(style="whitegrid")
  g = sns.catplot(
      data=df, kind="bar",
      x="in_sim", y="out_sim", hue="model",
      col="area", ci="sd",# palette="muted",
      height=5, aspect=1.2
  )
  g.set_axis_labels("Similaridade de Entrada", "Similaridade de Saída")
  g.set_titles("Área: {col_name}")

  sns.despine(left=True)
  plt.show()
