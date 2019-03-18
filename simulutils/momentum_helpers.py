# -*- coding: utf-8 -*-
"""
Preprocessing helper functions
"""
import pandas as pd
import numpy as np
from finhelpers import endpoints

def avg_mom_score_rev(data_prc, momentums, endpoint_key='m'):
    endpoint = endpoints(data_prc.index, endpoint_key)
    prc = data_prc.ix[endpoint]
    
    scores = 0.
    for i in momentums:
        scores = np.where((prc - prc.shift(i)) > 1., 1., 0.) + scores
    
    ret_df = pd.DataFrame(scores / len(momentums), columns=prc.columns, index=prc.index)

    # return only valid data. Make NaN for invalid one
    return ret_df[prc.shift(max(momentums)).notnull()]

def avg_mom_score(data_prc, momentums, endpoint_key='m'):
    endpoint = endpoints(data_prc.index, endpoint_key)
    prc = data_prc.ix[endpoint]
    
    scores = 0.
    for i in momentums:
        scores = np.where(prc/prc.shift(i) > 1., 1., 0.) + scores
    
    ret_df = pd.DataFrame(scores / len(momentums), columns=prc.columns, index=prc.index)

    # return only valid data. Make NaN for invalid one
    return ret_df[prc.shift(max(momentums)).notnull()]

def avg_mom(data_prc, momentums, endpoint_key='m'):
    endpoint = endpoints(data_prc.index, endpoint_key)
    prc = data_prc.ix[endpoint]
    
    scores = 0.
    for i in momentums:
        scores = prc/prc.shift(i) + scores
    return pd.DataFrame(scores / len(momentums), columns=prc.columns, index=prc.index)[prc.shift(max(momentums)).notnull()]