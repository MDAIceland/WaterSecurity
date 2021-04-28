
def dropColumnHalf(df,inplace=True):
    df.dropna(thresh=len(df.index)/2, axis=1, inplace=inplace)