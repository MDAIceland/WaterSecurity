
def dropColumnHalf(df):
    df.dropna(thresh=len(df.index)/2, axis=1, inplace=True)

def fill_missing_with_column(df, into, fro):
    """
    Merges one column into the other filling null values of the into colun and removing the fro column
    """
    df[into] = df[into].combine_first(df[fro])
    df.drop([fro], axis=1, inplace = True)
