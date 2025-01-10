from labels import labels
import pandas as pd
from load_data import load_data
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pymer4.models import Lmer
from rpy2.robjects import r
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
from rpy2.robjects import Formula
import rpy2.robjects as ro

pandas2ri.activate()
lme4 = importr('lme4')
glmmTMB = importr('glmmTMB')


def glm(df):
  formula = "out_sim ~ area * model * in_sim + (1|trial)"
  # model = Lmer("out_sim ~ area * model + in_sim + (1|trial)", data=df, family="gaussian")
  # model = Lmer("out_sim ~ area * model + in_sim + (1|trial)", data=df, family="gamma")
  model = lme4.glmer(
      formula,
      data=df,
      family=r.Gamma(link="inverse")
  )
  model.fit()
  print(model.summary())
  return model


def r_glmm(df):
  r_df = pandas2ri.py2rpy(df)

  formula = Formula("out_sim ~ area * model * in_sim + (1|trial)")

  model = glmmTMB.glmmTMB(
      formula,
      data=r_df,
      family=r['beta_family']()
  )

  print(r['summary'](model))
  return model


def predict_with_model(model, df):
  r_df_new = pandas2ri.py2rpy(df)
  return r.predict(model, newdata=r_df_new, type="response")


def plot_predicted_vs_observed(df, model):
  # df['predicted'] = model.predict(df, skip_data_checks=True, verify_predictions=False)
  df['predicted'] = predict_with_model(model, df)

  # Plot the observed vs predicted values
  plt.figure(figsize=(8, 6))
  plt.scatter(df['out_sim'], df['predicted'], alpha=0.5)
  plt.plot([0, 1], [0, 1], '--', color='r', label='Previsto = Observado')
  plt.xlabel(f'{labels['out_sim']} observada')
  plt.ylabel(f'{labels['out_sim']} prevista')
  plt.title('Valores Observados vs Previstos')
  plt.legend()
  # plt.show()
  plt.savefig(f'figures/plot_predicted_vs_observed.png')
  plt.close()


def plot(model):
  model.plot_summary()
  # model.plot()
  plt.show()


def interaction_plot(df):
  plt.figure(figsize=(10, 6))
  g = sns.lmplot(x='in_sim', y='out_sim', hue='area', col='model', data=df, logistic=True)
  for ax in g.axes.flatten():
    ax.set_xlabel(labels['in_sim'])
    ax.set_ylabel(labels['out_sim'])
  g._legend.set_title(labels['area'])
  g.set_titles(f'{labels['model']}: {{col_name}}')
  plt.show()
  # plt.savefig(f'figures/plot_legal.png')
  # plt.close()


def plot_random_effects(model):
  random_effects = model.random_effects

  # Plotting random effects for each trial
  plt.figure(figsize=(8, 6))
  plt.plot(random_effects.index, random_effects.values, 'o', label='Random Effects')
  plt.xlabel('Trial')
  plt.ylabel('Random Effects')
  plt.title('Random Effects by Trial')
  plt.legend()
  plt.show()


def residuals_plot(df, model):
  # df['predicted'] = model.predict(df, skip_data_checks=True, verify_predictions=False)
  df['predicted'] = predict_with_model(model, df)
  df['residuals'] = df['out_sim'] - df['predicted']

  # Plotting residuals
  plt.figure(figsize=(8, 6))
  sns.residplot(x=df['predicted'], y=df['residuals'], lowess=True, line_kws={'color': 'red'})
  plt.title('Resíduos vs Valores Previstos')
  plt.xlabel('Valores Previstos')
  plt.ylabel('Resíduos')
  # plt.show()
  plt.savefig(f'figures/plot_residuals.png')
  plt.close()


if __name__ == '__main__':
  df = load_data()

  model = r_glmm(df)

  # plot(model)
  # plot_random_effects(model)

  plot_predicted_vs_observed(df, model)
  residuals_plot(df, model)
  # interaction_plot(df)
