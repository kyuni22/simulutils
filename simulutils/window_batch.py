# -*- coding: utf-8 -*-
"""
Copyright (C) 2014-2016, KH LLC.
All rights reserved.
"""

class window_batch:
    '''
    The class instance keeps time series data (ts_data) with cursor to return slice of time series data set by 'batch_size'

    Todos
    Class variable is public now. it should be changed to private.
    '''
    def __init__(self, ts_data):
        '''
        Initialize with time series data and set cursor to 0
        :param ts_data: DataFrame with datetime index and Multi-columns which have security name and features.
                        Features level is 0, security name level is 1
        '''
        self.ts_data = ts_data
        self.cursor = 0
        self.max_cursor = ts_data.shape[0]

    def next_n_batch(self, batch_size, stack_flag=True):
        '''
        Return 'batch_size' window size data from current cursor
        :param batch_size: window size to return
        :param stack_flag: if stack_flag is True, returned data ecurity name will be on index. Otherwise no data structure change.
        :return: batch_size time series.
        '''
        from_cursor = self.cursor
        self.cursor = self.cursor + batch_size
        if stack_flag:
            return self.ts_data.iloc[from_cursor:self.cursor, :].stack()
        else:
            return self.ts_data.iloc[from_cursor:self.cursor, :]

    def last_n_batch(self, batch_size, stack_flag=True):
        '''
        Return last 'batch_size' window size data
        :param batch_size: window size to return
        :param stack_flag: if stack_flag is True, returned data ecurity name will be on index. Otherwise no data structure change.
        :return: batch_size time series.
        '''
        from_cursor = self.max_cursor - batch_size

        if stack_flag:
            return self.ts_data.iloc[from_cursor:self.max_cursor, :].stack()
        else:
            return self.ts_data.iloc[from_cursor:self.max_cursor, :]

    def reset_cursor(self, cursor_index):
        '''
        Reset cursor to 'cursor_index'
        :param cursor_index: to where to reset self.cursor
        '''
        self.cursor = cursor_index

    def check_window(self, n):
        '''
        Return True if 'n' more window data exist from current cursor
        :param n: data size to check
        '''
        return self.max_cursor >= self.cursor + n