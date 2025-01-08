from load_data import load_data
import statsmodels.api as sm
import statsmodels.formula.api as smf

if __name__ == '__main__':
  df = load_data()
  formula = 'out_sim ~ model + area + model:area'
  glm = smf.glm(formula=formula, data=df, family=sm.families.Gaussian()).fit()
  print(glm.summary())

