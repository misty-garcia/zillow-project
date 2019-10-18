import aquire

def rename_columns(df):
    df = df.rename(columns = {"bathroomcnt": "bathrooms", "bedroomcnt":"bedrooms", "calculatedfinishedsquarefeet":"squarefeet", "taxvaluedollarcnt":"taxvalue"})
    return df

def remove_zeros(df):
    df= df [df.bathrooms != 0] 
    df= df [df.bedrooms != 0] 
    df= df [df.squarefeet != 0] 
    df= df [df.taxvalue != 0]
    return df 
