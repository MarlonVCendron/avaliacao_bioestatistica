from load_data import load_data
import statsmodels.api as sm
import statsmodels.formula.api as smf


if __name__ == '__main__':
  df = load_data()
  formula = 'out_sim ~ model + area + model:area'
  # formula = 'out_sim ~ model * area * in_sim'
  # formula = 'out_sim ~ model * area'
  # formula = 'out_sim ~ model * area + model:in_sim'
  glm = smf.glm(formula=formula, data=df, family=sm.families.Gaussian()).fit()
  print(glm.summary())

