import aquire

def clean_zillow(df):
    df = df [df.bedrooms != 0]
    df = df [df.bathrooms != 0]
    df = df.dropna()
    df = df.astype({"counties":"category", "year":"category"})
    return df

