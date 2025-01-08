import pandas as pd
from load_data import load_data
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns



def glmm(df):
    from pymer4.models import Lmer


    model = Lmer("out_sim ~ in_sim + area * model + (1|trial)", data=df, family="beta")

    model.fit()
    print(model.summary())


def plot_predicted_vs_observed(df, model):
    # Predict the values based on the model
    df['predicted'] = model.predict(df)

    # Plot the observed vs predicted values
    plt.figure(figsize=(8, 6))
    plt.scatter(df['out_sim'], df['predicted'], alpha=0.5)
    plt.plot([0, 1], [0, 1], '--', color='r', label='Perfect fit')  # Ideal 1:1 line
    plt.xlabel('Observed out_sim')
    plt.ylabel('Predicted out_sim')
    plt.title('Observed vs Predicted out_sim')
    plt.legend()
    plt.show()



def plot_fixed_effects(model):
    # Extract fixed effects from the model
    fixed_effects = model.coefs_

    # Create a plot for the fixed effects
    plt.figure(figsize=(8, 6))
    sns.barplot(x=fixed_effects.index, y=fixed_effects.values)
    plt.title('Fixed Effects Estimates')
    plt.xlabel('Variables')
    plt.ylabel('Coefficient Estimate')
    plt.xticks(rotation=45)
    plt.show()



def interaction_plot(df):
    # Plotting the interaction between 'area', 'model', and 'in_sim'
    plt.figure(figsize=(10, 6))
    sns.lmplot(x='in_sim', y='out_sim', hue='area', col='model', data=df, logistic=True)
    plt.title('Interaction between Area and Model on out_sim')
    plt.show()



def plot_random_effects(model):
    random_effects = model.random_effects_

    # Plotting random effects for each trial
    plt.figure(figsize=(8, 6))
    plt.plot(random_effects.index, random_effects.values, 'o', label='Random Effects')
    plt.xlabel('Trial')
    plt.ylabel('Random Effects')
    plt.title('Random Effects by Trial')
    plt.legend()
    plt.show()



def residuals_plot(df, model):
    df['predicted'] = model.predict(df)
    df['residuals'] = df['out_sim'] - df['predicted']
    
    # Plotting residuals
    plt.figure(figsize=(8, 6))
    sns.residplot(x=df['predicted'], y=df['residuals'], lowess=True, line_kws={'color': 'red'})
    plt.title('Residuals Plot')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.show()

if __name__ == '__main__':
    df = load_data() 

    glmm(df) 
