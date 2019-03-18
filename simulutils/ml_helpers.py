# -*- coding: utf-8 -*-
"""

"""
import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.metrics import log_loss, accuracy_score
from sklearn.model_selection._split import _BaseKFold
from sklearn.externals import joblib

import bt

import sys
if sys.version_info[0] < 3:
    from window_batch import window_batch
    from finhelpers import qcut_signal
else:
    from .window_batch import window_batch
    from .finhelpers import qcut_signal

def make_target_weight_backtest(stock_data, signal, name, progress_bar=True):
    s = bt.Strategy(name, [bt.algos.bt.algos.RunOnDate(*signal.index),
                                bt.algos.bt.algos.WeighTarget(signal),
                                bt.algos.Rebalance()])
    return bt.Backtest(s, stock_data, initial_capital=10.**9, progress_bar=progress_bar)

def make_backtest_by_signal(stock_data, signal, name, progress_bar=False):
    s = bt.Strategy(name, [bt.algos.bt.algos.RunOnDate(*signal.index),
                                        bt.algos.bt.algos.SelectWhere(signal),
                                        bt.algos.WeighEqually(),
                                        bt.algos.Rebalance()])
    return bt.Backtest(s, stock_data, initial_capital=10.**9, progress_bar=progress_bar)

def sig_to_weight(sig_series, long_sig, short_sig, weight):
    long_count = (sig_series == long_sig).sum()
    short_count = (sig_series == short_sig).sum()
    
    if (long_count != 0) & (short_count != 0):
        sig_series.loc[(sig_series != long_sig) & (sig_series != short_sig)] = np.NaN
        sig_series.loc[sig_series == long_sig] = weight / long_count
        sig_series.loc[sig_series == short_sig] = -1. * weight / short_count
    else:
        sig_series.loc[:] = np.NaN
    
    return sig_series.fillna(0.)

def long_only_sig_to_weight(sig_series, sig, weight):
    long_count = (sig_series == sig).sum()
    
    if (long_count != 0):
        sig_series.loc[(sig_series != sig)] = np.NaN
        sig_series.loc[sig_series == sig] = weight / long_count
    else:
        sig_series.loc[:] = np.NaN
    
    return sig_series.fillna(0.)

def strat_settings_from_ml_prob(s_name, test_prob, cut_num, bt_fromdate, df_prc, shift_flag=False, ls_only=False, \
                                trade_cost=0., target_weight=1.0, ls_target_weight=0.5):

    sig_data = qcut_signal(test_prob, cut_num, test_prob.index[0])
    sig_data = sig_data.pivot(index='tdate', columns='code', values='value')
    sig_data.index = pd.to_datetime(sig_data.index)

    long_sig = cut_num*1. - 1.
    short_sig = 0.

    # do below if you want to check lagging effect
    if shift_flag:
        sig_data = sig_data.reindex(df_prc.index).shift(1).dropna(axis=0, how='all')

    strats = []
    
    if not ls_only:
        for signal in range(0, int(max(long_sig, short_sig))+1):
            weight_sig_data = sig_data.copy().apply(long_only_sig_to_weight, axis=1, args=(signal, target_weight)).loc[bt_fromdate:]
            s = bt.Strategy(s_name+'_'+str(signal), [bt.algos.RunOnDate(*weight_sig_data.index),
                                            bt.algos.WeighTarget(weight_sig_data),
                                            bt.algos.Rebalance()])
            strats.append(bt.Backtest(s, df_prc.loc[weight_sig_data.index[0]:].fillna(method='ffill'), initial_capital=10.**9, \
                                  commissions=lambda q, p: abs(q*p)*trade_cost))

    weight_sig_data = sig_data.copy().apply(sig_to_weight, axis=1, args=(long_sig, short_sig, ls_target_weight)).loc[bt_fromdate:]
    s = bt.Strategy(s_name+'_ls', [bt.algos.RunOnDate(*weight_sig_data.index),
                                        bt.algos.WeighTarget(weight_sig_data),
                                        bt.algos.Rebalance()])
    strats.append(bt.Backtest(s, df_prc.loc[weight_sig_data.index[0]:].fillna(method='ffill'), initial_capital=10.**9, \
                              commissions=lambda q, p: abs(q*p)*trade_cost))

    return strats, weight_sig_data

def comb_strat_settings_from_ml_prob(test_prob_dict, cut_num_dict, bt_fromdate, df_prc, shift_flag=False, comb_only=False, \
                                    trade_cost=0., ls_target_weight=0.5):
    weight_sig_data_dict = {}
    strats = []
    
    fsets = list(test_prob_dict.keys())
    
    weight_divisor = 1. / len(fsets)
    allidx = list(test_prob_dict[fsets[0]].index)
    codesToLoad = list(test_prob_dict[fsets[0]].columns)
    comb_sig = pd.DataFrame(0., index=allidx, columns=codesToLoad).loc[bt_fromdate:]

    for fset in fsets:
        print(fset)
        sig_data = qcut_signal(test_prob_dict[fset], cut_num_dict[fset], test_prob_dict[fset].index[0])
        sig_data = sig_data.pivot(index='tdate', columns='code', values='value')
        sig_data.index = pd.to_datetime(sig_data.index)
        long_sig = cut_num_dict[fset]*1. - 1.
        short_sig = 0.

        # do below if you want to check lagging effect
        if shift_flag:
            sig_data = sig_data.reindex(df_prc.index).shift(1).dropna(axis=0, how='all')

        weight_sig_data_dict[fset] = sig_data.copy().apply(sig_to_weight, axis=1, args=(long_sig, short_sig, ls_target_weight)).loc[bt_fromdate:]

        if not comb_only:
            s = bt.Strategy(fset, [bt.algos.RunOnDate(*weight_sig_data_dict[fset].index),
                                                bt.algos.WeighTarget(weight_sig_data_dict[fset]),
                                                bt.algos.Rebalance()])
            strats.append(bt.Backtest(s, df_prc.loc[weight_sig_data_dict[fset].index[0]:].fillna(method='ffill'), initial_capital=10.**9, \
                                      commissions=lambda q, p: abs(q*p)*trade_cost))

        tmp_sig_data = weight_sig_data_dict[fset].loc[bt_fromdate:] * weight_divisor
        comb_sig = comb_sig.add(tmp_sig_data.fillna(0.), fill_value=0.)

    weight_sig_data_dict['comb'] = comb_sig #[df_prc.loc[comb_sig.index, comb_sig.columns].notnull()]
    weight_sig_data_dict['comb'].fillna(0, inplace=True)

    s = bt.Strategy('comb', [bt.algos.RunOnDate(*weight_sig_data_dict['comb'].index),
                                        bt.algos.WeighTarget(weight_sig_data_dict['comb']),
                                        bt.algos.Rebalance()])
    strats.append(bt.Backtest(s, df_prc.loc[weight_sig_data_dict['comb'].index[0]:].fillna(method='ffill').fillna(0), initial_capital=10.**9, \
                              commissions=lambda q, p: abs(q*p)*trade_cost))    
    
    return strats, weight_sig_data_dict

def run_ml_model(ml_data, model_obj, startdate, train_days, test_days, \
                 target_feature, feature_set, filter_feature, \
                 train_flag, model_type, model_name='', \
                 model_save_flag=False, cat_features_list=None, model_save_path='./models/'):
    
    # Set window batch and params
    wb = window_batch(ml_data.loc[startdate:])

    reset_cursor = 0
    wb.reset_cursor(reset_cursor)
    
    test_signals = []

    # while window data is available
    while(wb.check_window(train_days + 1)):
        # Get train data
        train_XY = wb.next_n_batch(train_days)
        Y_train = train_XY.loc[:, target_feature].dropna()
        X_train = train_XY.loc[Y_train.index, feature_set].fillna(-1).values
        # Get test data
        test_XY = wb.next_n_batch(test_days)
        Y_test = test_XY.loc[:, filter_feature].dropna() # use one column to dropnas
        X_test = test_XY.loc[Y_test.index, feature_set].fillna(-1).values

        predict_start_date = test_XY.unstack().index[0].strftime('%Y%m%d')
        print('predict start from', predict_start_date)
        
        # run model
        if train_flag:
            if cat_features_list is not None:
                if (model_type == 'cb') | (model_type == 'cbr'):
                    model_obj.train(X_train, Y_train.dropna().values, cat_features=cat_features_list)
                elif model_type == 'lgb':
                    model_obj.train(X_train, Y_train.dropna().values, cat_feature=cat_features_list)
            else:
                model_obj.train(X_train, Y_train.dropna().values)
            
            # Saving Model
            if model_save_flag:
                if model_type == 'sklearn':
                    joblib.dump(model_obj.clf, model_save_path+model_name+'_'+predict_start_date)
                elif (model_type == 'cb') | (model_type == 'cbr') | (model_type == 'lgb'):
                    model_obj.clf.save_model(model_save_path+model_name+'_'+predict_start_date)
        else:
            if model_type == 'sklearn':
                model_obj.clf = joblib.load(model_save_path+model_name+'_'+predict_start_date)
            elif (model_type == 'cb') | (model_type == 'cbr'):
                model_obj.clf.load_model(model_save_path+model_name+'_'+predict_start_date)
            elif (model_type == 'lgb'):
                model_obj.load_model(model_save_path+model_name+'_'+predict_start_date)
            else:
                print('TODO: raise specify model type error')
                
        # predict & save signal
        if (model_type == 'sklearn') | (model_type == 'cb'):
            test_signals.append(pd.Series(model_obj.predict_proba(X_test)[:,1], index=Y_test.index).unstack())
        elif (model_type == 'lgb') | (model_type == 'cbr'):
            test_signals.append(pd.Series(model_obj.predict(X_test), index=Y_test.index).unstack())

        # reset cursor
        reset_cursor = reset_cursor + test_days
        wb.reset_cursor(reset_cursor)
        
    test_prob = pd.concat(test_signals)
    test_prob.columns.name = 'code'
    
    return test_prob

class PurgedKFold(_BaseKFold):
    '''
    Extend KFold Class to work with labels that span intervals
    The train is purged of observations overlapping test-label intervals
    Test set is assumed contiguous (shuffle=False), w/o traing samples in between
    
    Originally from AMF, but exclude purged part and pctEmbargo. just use split part.
    '''
    def __init__(self, n_splits=3, t1=None, pctEmbargo=0.):
        #if not isinstance(t1, pd.Series):
        #    raise ValueError('Label Through Dates must be a pd.Series')
        super(PurgedKFold, self).__init__(n_splits, shuffle=False, random_state=None)
        self.t1 = t1
        self.pctEmbargo = pctEmbargo
    
    def split(self, X, y=None, groups=None):
        #if (X.index == self.t1.index).sum() != len(self.t1):
        #    raise ValueError('X and ThruDateValues must have the same index')
        
        indices = np.arange(X.shape[0])
        mbrg = int(X.shape[0] * self.pctEmbargo)
        test_starts = [(i[0], i[-1]+1) for i in np.array_split(np.arange(X.shape[0]), self.n_splits)]
        
        for i, j in test_starts:
            # originally, below was purged and pctEmbargo part, but now simply implemented.
            test_indices = indices[i:j]
            train_indices = np.concatenate((indices[:i], indices[j:]))
            yield train_indices, test_indices

def wb_cvScore(clf, XY, featNames, targetName, sample_weight=None, scoring='neg_log_loss', t1=None, cv=None, cvGen=None, pctEmbargo=0.):
    if scoring not in ['neg_log_loss', 'accuracy']:
        raise Exception('wrong scoring method')
    if cvGen is None:
        cvGen = PurgedKFold(n_splits=cv, t1=t1, pctEmbargo=pctEmbargo) # purged
    
    score = []
    for train, test in cvGen.split(X=XY):
        trainXY = XY.iloc[train, :].stack()
        trainy = trainXY.loc[:, targetName].dropna()
        trainX = trainXY.loc[trainy.index, featNames].fillna(-1)
        testXY = XY.iloc[test, :].stack()
        testy = testXY.loc[:, targetName].dropna()
        testX = testXY.loc[testy.index, featNames].fillna(-1)
        fit = clf.fit(X=trainX, y=trainy, sample_weight=None)
        if scoring == 'neg_log_loss':
            prob = fit.predict_proba(testX)
            score_ = -log_loss(testy, prob, sample_weight=None, labels=clf.classes_)
        else:
            pred = fit.predict(testX)
            score_ = accuracy_score(testy, pred, sample_weight=None)
        score.append(score_)
    
    return np.array(score)

# Mean Decrease Impurity (MDI)
def featImpMDI(fit, featNames):
    # feat importance based on IS mean impurity reduction
    df0 = {i: tree.feature_importances_ for i, tree in enumerate(fit.estimators_)}
    df0 = pd.DataFrame.from_dict(df0, orient='index')
    df0.columns = featNames
    df0 = df0.replace(0, np.nan) # becasue max_features = 1.
    imp = pd.concat({'mean': df0.mean(), 'std': df0.std()*df0.shape[0]**-.5}, axis=1)
    imp /= imp['mean'].sum()
    return imp

# Mean Decrease Accuracy
def featImpMDA(clf, XY, featNames, targetName, cv, sample_weight=None, t1=None, \
               pctEmbargo=0., scoring='neg_log_loss'):
    # feat importance based on OOS score reduction
    if scoring not in ['neg_log_loss', 'accuracy']:
        raise Exception('wrong scoring method')
    cvGen = PurgedKFold(n_splits=cv, t1=t1, pctEmbargo=pctEmbargo) # purged cv
    scr0, scr1 = pd.Series(), pd.DataFrame(columns = featNames)
    
    for i, (train, test) in enumerate(cvGen.split(X=XY)):
        trainXY = XY.iloc[train, :].stack()
        trainy = trainXY.loc[:, targetName].dropna()
        trainX = trainXY.loc[trainy.index, featNames].fillna(-1)
        testXY = XY.iloc[test, :].stack()
        testy = testXY.loc[:, targetName].dropna()
        testX = testXY.loc[testy.index, featNames].fillna(-1)
        
        X0, y0, w0 = trainX, trainy, None
        X1, y1, w1 = testX, testy, None
        fit = clf.fit(X=X0, y=y0, sample_weight=None)
        
        if scoring == 'neg_log_loss':
            prob = fit.predict_proba(X1)
            scr0.loc[i] = -log_loss(y1, prob, sample_weight = None, labels = clf.classes_)
        else: # accuracy
            pred = fit.predict(X1)
            scr0.loc[i] = accuracy_score(y1, pred, sample_weight = None)
        
        for j in featNames:
            X1_ = X1.copy(deep=True)
            np.random.shuffle(X1_[j].values) # permutation of a single columns
            if scoring == 'neg_log_loss':
                prob = fit.predict_proba(X1_)
                scr1.loc[i, j] = -log_loss(y1, prob, sample_weight=None, labels=clf.classes_)
            else: # accuracy
                pred = fit.predict(X1_)
                scr1.loc[i, j] = accuracy_score(y1, pred, sample_weight=None)
    
    imp = (-1*scr1).add(scr0, axis=0)

    if scoring == 'neg_log_loss':
        imp = imp/(-1*scr1)
    else:
        imp = imp / (1.-scr1)

    imp = pd.concat({'mean':imp.mean(), 'std': imp.std()*imp.shape[0]**-.5}, axis=1)
    return imp, scr0.mean()