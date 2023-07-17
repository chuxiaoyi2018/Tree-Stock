#!/usr/bin/env python
# coding: utf-8

# In[1]:


from get_stock_data import Downloader, mkdir
import os
import argparse
from tqdm import tqdm
import pandas as pd
import numpy as np
import catboost as cb
import datetime
import functools
from collections import defaultdict, Counter

from utils import reduce_mem

def update_dataset():
#     downloader = Downloader(params.raw_train_path, 
#                             date_start=params.date_start, # start和 end都包含
#                             date_end=params.date_end, 
#                             frequency='d',
#                             header=False,
#                             mode='all')
#     downloader.run()

    def df_concat(csv_path):
        df_arr = []
        for file_path in tqdm(os.listdir(csv_path)):
            try:
                if file_path.split('.') != 'csv':pass
                temp = pd.read_csv(csv_path + '/' + file_path, engine='python')
                temp['intDate'] = temp.date.apply(lambda x: int(x.replace('-', '')))
                temp = temp[temp['intDate'] >= params.date_abandon].reset_index(drop=True)
                if len(temp)==0:continue
                df_arr.append(temp)
            except:
                pass
        return pd.concat(df_arr)


    train = df_concat(params.raw_train_path)
    train = train.drop('intDate', axis=1)
    assert params.date_end in train.date.unique()
    train.to_csv(params.train_path, index=False)
    return

class params:
    mode = 'inference'
    window_size = 14
    
    date_abandon = 20210101
    date_split = 20221111
    date_start = '2022-06-02'
    date_end = '2022-11-11' # 不能选择今天的日期，周末的日期
    
    raw_train_path = 'stockdata/d_train'
    raw_test_path = 'stockdata/d_test'
    train_path = 'stockdata/d_data/train.csv'
    test_path = 'stockdata/d_data/test.csv'
    industry_path = 'stockdata/stock_industry.csv'
    concept_path = 'stockdata/concept_df.csv'
    concept_hist_path = 'stockdata/concept_hist_df.csv'
    submit_path = f'submit/{date_end}.csv'    
    
    model_params = {'n_estimators':5000,
          'learning_rate': 0.05,
          'max_depth': 7,
          'early_stopping_rounds':1000,
          'loss_function':'MultiClass',
           'classes_count':3,
          'max_bin':512,
          'subsample':0.8,
          'bootstrap_type':'Poisson',
          'random_seed':np.random.randint(0,2021)}
    
    
if __name__ == '__main__':
    update_dataset()