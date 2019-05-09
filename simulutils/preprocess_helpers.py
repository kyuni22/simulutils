# -*- coding: utf-8 -*-
"""
Preprocessing helper functions
"""
import pandas as pd
import numpy as np
from scipy.special import erfinv

def ts_sampling_fs(fs, ts_index):
    # convert to datetime and process for ts
    fs.index = pd.to_datetime(fs.index, format='%Y%m')
    fs.index = [x + pd.DateOffset(months=1) if x.month == 3 else x for x in list(fs.index)]

    # add new period
    if fs.index[-1].month == 12:
        fs.loc[fs.index[-1] + pd.DateOffset(months=4), :] = np.NaN
    elif fs.index[-1].month == 4:
        fs.loc[fs.index[-1] + pd.DateOffset(months=2), :] = np.NaN
    else:
        fs.loc[fs.index[-1] + pd.DateOffset(months=3), :] = np.NaN

    fs = fs.shift(1)

    return fs.reindex(ts_index, method='ffill')

def getWeights_FFD(d, thres):
    w, k=[1.], 1
    while True:
        w_ = -w[-1]/k*(d-k+1)
        if abs(w_) < thres:
            break
        w.append(w_)
        k += 1
    return np.array(w[::-1]).reshape(-1,1)

def fracDiff_FFD(series, d, thres=1e-5):
    # Constant width window (new solution)
    w = getWeights_FFD(d, thres)
    width, df = len(w)-1, {}

    for name in series.columns:
        seriesF, df_ = series[[name]].fillna(method='ffill').dropna(), pd.Series()
        for iloc1 in range(width, seriesF.shape[0]):
            loc0, loc1 = seriesF.index[iloc1-width], seriesF.index[iloc1]
            if not np.isfinite(series.loc[loc1, name]):
                continue # exclude NAs
            df_.loc[loc1]=np.dot(w.T, seriesF.loc[loc0:loc1])[0,0]
        df[name]=df_.copy(deep=True)
    df=pd.concat(df,axis=1)
    return df

def get_data_df(df_data, item_code):
    ret_df = df_data.loc(axis=1)[:, item_code]
    ret_df.columns = ret_df.columns.get_level_values(0)
    ret_df.columns.name = 'code'
    ret_df.index.name = 'tdate'
    return ret_df.applymap(lambda x: x if type(x) == type(1.) else x.replace(',','')).apply(pd.to_numeric)


def ts_scaling(df_data, how='standard', ex_cols=[]):
    '''
    Scaling ml data
    input:
    df_data: dataframe with time index and multi-index column with stock code and feature name
    how: how to scale data. there are 4 ways - standard, minmax, rank, robust
    ex_cols: list of column names to exclude from scaling
    '''
    new_df = []
    for i in df_data.index:
        df_slice = df_data.loc[i].unstack(0)

        # excluding ex_cols
        if len(ex_cols) > 0:
            ex_slice = df_slice[ex_cols]
            df_slice.drop(ex_cols, axis=1, inplace=True)
        else:
            ex_slice = pd.DataFrame(index=df_slice.index)

        # scaling data
        if how == 'rank':
            df_slice = df_slice.rank(axis=0, na_option='keep', pct=True)
        elif how == 'minmax':
            df_slice = (df_slice - np.nanmin(df_slice, axis=0)) / (
                        np.nanmax(df_slice, axis=0) - np.nanmin(df_slice, axis=0))
        elif how == 'standard':
            df_slice = (df_slice - np.nanmean(df_slice, axis=0)) / np.nanstd(df_slice, axis=0)
        elif how == 'robust':
            df_slice = (df_slice - np.nanquantile(df_slice, q=0.25, axis=0)) / (
                        np.nanquantile(df_slice, q=0.75, axis=0) - np.nanquantile(df_slice, q=0.25, axis=0))
        elif how == 'gaussrank':
            eps = 0.0001
            upper = 1 - eps
            lower = -1 + eps
            df_slice = erfinv(df_slice.rank(axis=0, na_option='keep', method='max', pct=True) * (upper - lower) - upper)

        df_slice = pd.concat([df_slice, ex_slice], axis=1)

        df_slice = df_slice.unstack()
        df_slice.name = i
        new_df.append(df_slice)
    return pd.concat(new_df, 1).T


def ts_transform(df_data, transformer, imputer):
    for i in df_data.index:
        to_transform = df_data.loc[i].unstack(0)
        origin_col = to_transform.columns
        origin_index = to_transform.index
        
        if to_transform.isnull().values.any():
            print("imputed " + str(i))
            to_transform = imputer.fit_transform(to_transform)
            
        transformed = pd.DataFrame(qt.fit_transform(to_transform), index=origin_index, columns=origin_col)
        df_data.loc[i] = transformed.unstack()
        
def ts_nan_transform(df_data, rank_flag=False, ex_cols=[]):
    for i in df_data.index:
        original_slice = df_data.loc[i].unstack(0)
        origin_col = original_slice.columns
        origin_index = original_slice.index
        
        cols_to_transform = [col for col in original_slice.columns if col not in ex_cols]
        to_transform = original_slice[cols_to_transform]
        
        if rank_flag:
            to_transform = to_transform.rank(axis=0, na_option='keep')
        
        scaler_ = (np.nanmax(to_transform, axis=0) - np.nanmin(to_transform, axis=0))
        min_ = np.nanmin(to_transform, axis=0)
        
        to_transform = (to_transform - min_) / scaler_
        
        if len(ex_cols) > 0.:
            transformed = pd.DataFrame(to_transform, index=origin_index, columns=cols_to_transform)
            transformed = transformed.merge(original_slice[ex_cols], left_index=True, right_index=True)
        else:
            transformed = pd.DataFrame(to_transform, index=origin_index, columns=origin_col)
        
        df_data.loc[i] = transformed.unstack()