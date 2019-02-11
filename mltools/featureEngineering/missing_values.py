import missingno as msno
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def get_missing_values(df, plot = True, plot_matrix = True, features_lim = None, figsize=(15, 10), dpi = 200):
    missing = df.isnull().sum() 
    missing_count = missing[missing.values != 0].sort_values(ascending=False)
    missing_perc = missing_count.apply(lambda x: (x/len(df))*100)
    missing_df = pd.DataFrame({'Missing Ratio' : missing_perc, 'Missing count': missing_count})
    if plot:
        f, ax = plt.subplots(figsize=figsize, dpi=dpi)
        plt.xticks(rotation='70')
        if features_lim:
            data = missing_df[:features_lim]
        else:
            data = missing_df
        sns.barplot(x=data.index, y='Missing Ratio', data = data, color=(0.20, 0.20, 0.20))
        plt.xlabel('Features', fontsize=15)
        plt.ylabel('Percent of missing values', fontsize=15)
        plt.title('Percent missing data by feature', fontsize=15)
    if plot_matrix:
        msno.matrix(df[missing_df.index])
    return missing_df