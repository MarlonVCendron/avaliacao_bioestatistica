import pandas as pd
from load_data import load_data


def trial_count(df):

    df.reset_index(drop=True, inplace=True)
    df.sort_values(by=['model', 'area'], inplace=True)

    # Initialize the trial counter
    df['trial'] = 1

    # Loop through the dataframe to assign trial numbers
    for i in range(1, len(df)):
        if (df.loc[i, 'model'] != df.loc[i-1, 'model'] or 
            df.loc[i, 'area'] != df.loc[i-1, 'area'] or 
            df.loc[i, 'in_sim'] != df.loc[i-1, 'in_sim']):
            df.loc[i, 'trial'] = 1
        else:
            # Otherwise, increment the trial number
            df.loc[i, 'trial'] = df.loc[i-1, 'trial'] + 1

    # Save the modified dataframe to a new CSV
    df.to_csv(r'C:\Users\pcost\Desktop\biostat\dados_trial.csv', index=False)

    print(df.head())  # Print the first few rows to check the result

if __name__ == '__main__':
    df = load_data ()
    trial_count (df)
