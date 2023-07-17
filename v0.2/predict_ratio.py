#!/usr/bin/env python
# coding: utf-8

# In[1]:


from get_stock_data import Downloader, mkdir
import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import catboost as cb

# 获取全部股票的日K线数据
mkdir('stockdata/d_data')
raw_train_path = 'stockdata/d_train'
raw_test_path = 'stockdata/d_test'
train_path = 'stockdata/d_data/train.csv'
test_path = 'stockdata/d_data/test.csv'
mode = 'train'


# In[2]:


if mode == 'debug':
    train = pd.read_csv(train_path, nrows=100000)
    test = pd.read_csv(test_path, nrows=100000)
else:
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)


def feature_engineer(train, test):
    train_len = len(train)
    data = pd.concat((train, test), sort=False).reset_index(drop=True)
    
    stock_industry = pd.read_csv("stock_industry.csv", encoding="gbk")
    from sklearn.preprocessing import LabelEncoder
    lbe = LabelEncoder()
    stock_industry['industry'] = lbe.fit_transform(stock_industry['industry'])
    data = pd.merge(data, stock_industry[['code', 'industry']], how='left', on='code')
    
#     for name in tqdm(['close', 'volume', 'amount', 'turn', 'pctChg', 'pbMRQ']):
#         for day in ['7', '30']:
#             rolling = data.groupby['code'][f'{name}'].rolling(window=int(day), center=False)
#             data[f'{name}_rolling_{day}_mean'] = rolling.mean().reset_index(drop=True)
#             data[f'{name}_rolling_{day}_max'] = rolling.max().reset_index(drop=True)
#             data[f'{name}_rolling_{day}_min'] = rolling.min().reset_index(drop=True)
#             data[f'{name}_rolling_{day}_sum'] = rolling.sum().reset_index(drop=True)
#             data[f'{name}_rolling_{day}_median'] = rolling.median().reset_index(drop=True)
#             data[f'{name}_rolling_{day}_skew'] = rolling.skew().reset_index(drop=True)
#             data[f'{name}_rolling_{day}_kurt'] = rolling.kurt().reset_index(drop=True)
#             data[f'{name}_rolling_{day}_std'] = rolling.std().reset_index(drop=True)
#             data[f'{name}_rolling_{day}_mad'] = rolling.mad()
#             data[f'{name}_rolling_{day}_autocorr1'] = rolling.autocorr(1)
#             data[f'{name}_rolling_{day}_autocorr2'] = rolling.autocorr(2)
    
            
    
    return data.iloc[:train_len].reset_index(drop=True), data.iloc[train_len:].reset_index(drop=True)


# In[ ]:


train, test = feature_engineer(train, test)

# f_train_path = 'stockdata/d_data/f_train_debug.csv'
# f_test_path = 'stockdata/d_data/f_test_debug.csv'
# train.to_csv(f_train_path, index=False)
# test.to_csv(f_test_path, index=False)


# In[ ]:


train['label'] = train.groupby('code').close.transform(lambda x:(x - x.shift(-14)) / (x + 1e-7) )
test['label'] = test.groupby('code').close.transform(lambda x:(x - x.shift(-14)) / (x + 1e-7) )

train = train.dropna(subset = ['label'], inplace=False)
test = test.dropna(subset = ['label'], inplace=False)

train = train.replace(np.nan, 0)
test = test.replace(np.nan, 0)

ycol = 'label'
feature_names = list(
    filter(lambda x: x not in [ycol, 'code', 'date', ''], train.columns))

# print(feature_names)


# In[ ]:


quantile_30, quantile_70 = train.label.quantile([0.3, 0.7]).values


# In[ ]:


def label_quantile(x):
    if x<quantile_30:
        return 0
    elif x<quantile_70:
        return 1
    else:
        return 2


# In[ ]:


train.label = train.label.apply(label_quantile)
test.label = test.label.apply(label_quantile)


params = {'n_estimators':5000,
      'learning_rate': 0.05,
      'max_depth': 7,
      'early_stopping_rounds':1000,
      'loss_function':'MultiClass',
       'classes_count':3,
      'max_bin':512,
#       'subsample':0.8,
#       'bootstrap_type':'Poisson',
      'random_seed':np.random.randint(0,2021)}

model = cb.CatBoostClassifier(eval_metric="AUC", task_type='CPU', **params)

X_train = train[feature_names]
Y_train = train[ycol]

X_val = test[feature_names]
Y_val = test[ycol]


cat_model = model.fit(X_train,
                      Y_train,
                      # eval_names=['train', 'valid'],
                      # eval_set=[(X_train, Y_train), (X_val, Y_val)],
                      eval_set=(X_val, Y_val),
                      plot=False,
                      verbose=500)


df_importance = pd.DataFrame({
    'column': feature_names,
    'importance': cat_model.feature_importances_,
})

# cat_model.save_model(f'cb_{frequency}.model')


# In[ ]:


# ?model.fit


# In[ ]:


print(df_importance)


# In[ ]:


cat_model.save_model(f'model/cb_next_2week.model')


# In[ ]:




