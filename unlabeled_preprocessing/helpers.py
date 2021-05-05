from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import pandas as pd

def dropColumnHalf(df):
    """
    Removes columns where the number of missig values is 50% or more
    """
    df.dropna(thresh=len(df.index)/2, axis=1, inplace=True)

def fill_missing_with_column(df, into, fro):
    """
    Merges one column into the other filling null values of the into colun and removing the fro column
    """
    df[into] = df[into].combine_first(df[fro])
    df.drop([fro], axis=1, inplace = True)

    
def impute_df(df, max_iter=10, verbose=0):
    """
    Imputes a df and returns a dataframe with the original and imputed values
    """
    imp = IterativeImputer(max_iter=max_iter, verbose=verbose)
    imp.fit_transform(df)
    imputed_df = imp.transform(df)
    return pd.DataFrame(imputed_df, columns=df.columns,index=df.index)

def print_missing_percentages(df):
    """
    Max, min and mean number of missing values for the columns
    """
    percent_missing = df.isnull().sum() * 100 / len(df)
    max_missing = percent_missing.max()
    min_missing = percent_missing.min()
    mean_missing = percent_missing.mean()
    print("Max, min and mean number of missing values for the columns")
    print("Max:", max_missing,'%')
    print("Min:", min_missing,'%')
    print("Mean:", mean_missing,'%')
    return min_missing, max_missing

def find_all_integer_columns(df):
    """
    Returns an array of all colums that contain only integers or null values
    """
    integer_columns = df.applymap(lambda x: int(x)==x if pd.notnull(x) else x).prod().values.astype(bool)
    return df.columns[integer_columns].values

