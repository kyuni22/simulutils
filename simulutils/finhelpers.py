# -*- coding: utf-8 -*-
"""

"""

import numpy as np
import pandas as pd

# Data change helpers
def endpoints(dates, period='m'):
    """

    :param dates:
    :param period:
    :return:
    """
    if isinstance(period, int):
        dates = [dates[i] for i in range(0, len(dates), period)]
    elif period == 'w':
        dates = [dates[i - 1] for i in range(1, len(dates)) \
                if dates[i].dayofweek < dates[i - 1].dayofweek \
                or (dates[i] - dates[i - 1]).days > 7] + list([dates[-1]])
    else:
        if period == 'm':
            months = 1
        elif period == 'q':
            months = 3
        elif period == 'b':
            months = 6
        elif period == 'y':
            months = 12

        e_dates = [dates[i - 1] for i in range(1, len(dates)) \
                   if dates[i].month > dates[i - 1].month \
                   or dates[i].year > dates[i - 1].year] + list([dates[-1]])
        dates = [e_dates[i] for i in range(0, len(e_dates), months)]

    return dates


# Backtesting helpers
def qcut_signal(toTest, cut_number, startDate):
    """

    :param toTest:
    :param cut_number:
    :param startDate:
    :return:
    """
    quantile_values = []

    for date in toTest[startDate:].index:
        try:
            tmpDF = toTest.ix[date]
    #        tmpDF = tmpDF[~((tmpDF - tmpDF.mean()).abs() > 5.0 * tmpDF.std())]
            tmpDF = pd.qcut(tmpDF, cut_number, labels=False).sort_values(ascending=True).to_frame()
            tmpDF.reset_index(inplace=True)
            quantile_values.append(pd.melt(tmpDF, id_vars=['code'], var_name='tdate'))
        except ValueError: # for unique qcut value
            print(date)
    return pd.concat(quantile_values, axis=0)

def get_weights_by_signal(signal_data, signal):
    """

    :param signal_data:
    :param signal:
    :return:
    """
    signal_data = signal_data.pivot(index='tdate', columns='code', values='value')
    weights = signal_data.applymap(lambda x: 1. if x == signal else 0.)
    counts = weights.sum(axis=1)
    return weights.div(counts, axis=0)

def groupby_backtest(returns, qcut_signal_data):
    """

    :param returns:
    :param qcut_signal_data:
    :return:
    """
    simul_data = qcut_signal_data.merge(returns, left_on=['tdate', 'code'], right_on=['tdate', 'code'])
    simul_data = simul_data[sorted(simul_data.columns)]
    simul_data.columns = ['code', 'tdate', 'sig', 'ret']
    grouped_simul_val = simul_data.groupby(['tdate', 'sig'])
    return grouped_simul_val.mean().unstack()