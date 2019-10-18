import aquire

def remove_zeros(df):
    return df.loc[(df!=0).all(axis=1)]

