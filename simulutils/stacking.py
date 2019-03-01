# -*- coding: utf-8 -*-
"""
Faron의 stacking-starter code의 일부를 사용함
https://www.kaggle.com/mmueller/stacking-starter

stacking util
"""
# utility
import pandas as pd
import numpy as np
#from sklearn.cross_validation import KFold

# algorithm
import xgboost as xgb
import lightgbm as lgbm

# metric
from sklearn.metrics import log_loss,mean_absolute_error,mean_squared_error, r2_score

class SklearnWrapper(object):
    def __init__(self, clf, params=None, **kwargs):
        # kwargs setting
        y_value_log = kwargs.get('y_value_log', False)

        self.clf = clf(**params)
        self.y_value_log = y_value_log

    def train(self, x_train, y_train, x_cross=None, y_cross=None):
        if self.y_value_log is True:
            self.clf.fit(x_train, np.log(y_train))
        else:
            self.clf.fit(x_train, y_train)

    def predict(self, x):
        if self.y_value_log is True:
            return np.exp(self.clf.predict(x))
        else:
            return self.clf.predict(x)

    def predict_proba(self, x):
        return self.clf.predict_proba(x)

class CatbWrapper(object):
    def __init__(self, clf, params=None, **kwargs):
        # kwargs setting
        y_value_log = kwargs.get('y_value_log', False)

        self.clf = clf(**params)
        self.y_value_log = y_value_log

    def train(self, x_train, y_train, cat_features=None):
        if self.y_value_log is True:
            self.clf.fit(x_train, np.log(y_train), cat_features=cat_features)
        else:
            self.clf.fit(x_train, y_train, cat_features=cat_features)

    def predict(self, x):
        if self.y_value_log is True:
            return np.exp(self.clf.predict(x))
        else:
            return self.clf.predict(x)

    def predict_proba(self, x):
        return self.clf.predict_proba(x)    
    
class XgbWrapper(object):
    def __init__(self, params=None, **kwargs):
        seed = kwargs.get('seed', 0)
        num_rounds = kwargs.get('num_rounds', 1000)
        early_stopping = kwargs.get('ealry_stopping', 100)
        eval_function = kwargs.get('eval_function', None)
        verbose_eval = kwargs.get('verbose_eval', 100)
        y_value_log = kwargs.get('y_value_log', False)
        base_score = kwargs.get('base_score', False)
        feval_maximize = kwargs.get('maximize', False)

        if 'silent' not in params:
            params['silent'] = 1

        self.param = params
        self.param['seed'] = seed
        self.num_rounds = num_rounds
        self.early_stopping = early_stopping

        self.eval_function = eval_function
        self.verbose_eval = verbose_eval
        self.y_value_log = y_value_log
        self.base_score = base_score
        self.feval_maximize = feval_maximize

    def train(self, x_train, y_train, x_cross=None, y_cross=None):
        need_cross_validation = True
        if x_cross is None:
            need_cross_validation = False

        if self.base_score is True:
            y_mean = np.mean(y_train)
            self.param['base_score'] = y_mean

        if self.y_value_log is True:
            y_train = np.log(y_train+1)
            if need_cross_validation is True:
                y_cross = np.log(y_cross+1)

        if need_cross_validation is True:
            dtrain = xgb.DMatrix(x_train, label=y_train)
            dvalid = xgb.DMatrix(x_cross, label=y_cross)
            watchlist = [(dtrain, 'train'), (dvalid, 'eval')]

            self.clf = xgb.train(self.param, dtrain, self.num_rounds, watchlist, feval=self.eval_function,
                                 early_stopping_rounds=self.early_stopping, maximize=self.feval_maximize,
                                 verbose_eval=self.verbose_eval)
        else:
            dtrain = xgb.DMatrix(x_train, label=y_train, silent= True)
            self.clf = xgb.train(self.param, dtrain, self.num_rounds)

    def predict(self, x):
        if self.y_value_log is True:
            return np.exp(self.clf.predict(xgb.DMatrix(x)))-1
        else:
            return self.clf.predict(xgb.DMatrix(x))

    def get_params(self):
        return self.param

class LgbmWrapper(object):
    def __init__(self, params=None, **kwargs):
        seed = kwargs.get('seed', 0)
        num_rounds = kwargs.get('num_rounds', 1000)
        early_stopping = kwargs.get('ealry_stopping', 100)
        eval_function = kwargs.get('eval_function', None)
        verbose_eval = kwargs.get('verbose_eval', 100)
        y_value_log = kwargs.get('y_value_log', False)
#        base_score = kwargs.get('base_score', False) does not exist in recent version
        feval_maximize = kwargs.get('maximize', False)

        self.param = params
        self.param['seed'] = seed
        self.num_rounds = num_rounds
        self.early_stopping = early_stopping

        self.eval_function = eval_function
        self.verbose_eval = verbose_eval
        self.y_value_log = y_value_log
#        self.base_score = base_score
        self.feval_maximize = feval_maximize

    def train(self, x_train, y_train, x_cross=None, y_cross=None, cat_feature='auto'):
        need_cross_validation = True
        if x_cross is None:
            need_cross_validation = False

        if isinstance(y_train, pd.DataFrame) is True:
            y_train = y_train[y_train.columns[0]]
            if need_cross_validation is True:
                y_cross = y_cross[y_cross.columns[0]]

#        if self.base_score is True:
#            y_mean = np.mean(y_train)
#            self.param['init_score '] = y_mean does not exist in recent version

        if self.y_value_log is True:
            y_train = np.log(y_train+1)
            if need_cross_validation is True:
                y_cross = np.log(y_cross+1)

        if need_cross_validation is True:
            dtrain = lgbm.Dataset(x_train, label=y_train, categorical_feature=cat_feature, silent=True)
            dvalid = lgbm.Dataset(x_cross, label=y_cross, categorical_feature=cat_feature, silent=True)
            self.clf = lgbm.train(self.param, train_set=dtrain, num_boost_round=self.num_rounds, valid_sets=dvalid,
                                  feval=self.eval_function, categorical_feature=cat_feature, early_stopping_rounds=self.early_stopping,
                                  verbose_eval=self.verbose_eval)
        else:
            dtrain = lgbm.Dataset(x_train, label=y_train, categorical_feature=cat_feature, silent= True)
            self.clf = lgbm.train(self.param, dtrain, self.num_rounds, categorical_feature=cat_feature)

    def predict(self, x):
        if self.y_value_log is True:
            return np.exp(self.clf.predict(x))-1 # erase "num_iteration=self.clf.best_iteration" parameter since it already has function to use best_iter
        else:
            return self.clf.predict(x) # erase "num_iteration=self.clf.best_iteration" parameter since it already has function to use best_iter

    def get_params(self):
        return self.param

    def load_model(self, model_path):
        self.clf = lgbm.Booster(model_file=model_path)


class KerasWrapper(object):
    def __init__(self, model, callback, **kwargs):
        self.model = model
        self.callback = callback

        epochs = kwargs.get('epochs', 300)
        batch_size = kwargs.get('batch_size', 30)
        shuffle = kwargs.get('shuffle', True)
        verbose_eval = kwargs.get('verbose_eval',1)

        self.epochs = epochs
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.verbose_eval = verbose_eval

    def train(self, x_train, y_train, x_cross=None, y_cross=None):
        need_cross_validation = True

        if x_cross is None:
            need_cross_validation = False

        if isinstance(y_train, pd.DataFrame) is True:
            y_train = y_train[y_train.columns[0]]
            if need_cross_validation is True:
                y_cross = y_cross[y_cross.columns[0]]

        if isinstance(x_train, pd.DataFrame) is True:
            x_train = x_train.values
            if need_cross_validation is True:
                x_cross = x_cross.values

        if need_cross_validation is True:
            self.history = self.model.fit(x_train, y_train,
                                          nb_epoch=self.epochs,
                                          batch_size = self.batch_size,
                                          validation_data=(x_cross, y_cross),
                                          verbose=self.verbose_eval,
                                          callbacks=self.callback,
                                          shuffle=self.shuffle)
        else:
            self.model.fit(x_train, y_train,
                           nb_epoch=self.epochs,
                           batch_size=self.batch_size,
                           verbose=self.verbose_eval,
                           callbacks=self.callback,
                           shuffle=self.shuffle)

    def predict(self, x):
        print(x.shape)
        if isinstance(x, pd.DataFrame) is True:
            x = x.values

        result = self.model.predict(x)
        return result.flatten()

    def get_params(self):
        return self.model.summary()