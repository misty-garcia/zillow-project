import pandas as pd

import util

def get_data(query, db):
    return pd.read_sql(query, util.get_url(db))


# def get_data(db):
#     query = """
#     SELECT bathroomcnt as bathrooms, 
#     bedroomcnt as bedrooms,
#     calculatedfinishedsquarefeet as squarefeet, 
#     taxvaluedollarcnt as taxvalue
#         FROM properties_2017
#         JOIN predictions_2017 USING (parcelid)
#         WHERE transactiondate BETWEEN "2017-05-01" AND "2017-06-31"
#             AND propertylandusetypeid in ("261")
#             AND calculatedfinishedsquarefeet is not null
#             AND NOT (finishedsquarefeet50 is not null AND finishedsquarefeet50 > calculatedfinishedsquarefeet)
#         """
#     return pd.read_sql(query, util.get_url(db))

