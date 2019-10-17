import pandas as pd

import util

def get_data(db):
    query = """
    SELECT bathroomcnt, bedroomcnt,calculatedfinishedsquarefeet, taxvaluedollarcnt, fips
    FROM properties_2017
    JOIN predictions_2017 USING (parcelid)
    WHERE transactiondate BETWEEN "2017-05-01" AND "2017-06-31"
	    AND propertylandusetypeid in ("261")
	    AND finishedsquarefeet12 is not null
        """
    return pd.read_sql(query, util.get_url(db))
