#!/usr/bin/env python
# coding: utf-8

# In[1]:


from get_stock_data import Downloader, mkdir
import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import catboost as cb
import datetime

# 获取全部股票的日K线数据
mkdir('stockdata/d_data')
raw_train_path = 'stockdata/d_train'
raw_test_path = 'stockdata/d_test'
train_path = 'stockdata/d_data/train.csv'
test_path = 'stockdata/d_data/test.csv'
mode = 'train'


# In[ ]:


if mode == 'debug':
    train = pd.read_csv(train_path, nrows=100000)
    test = pd.read_csv(test_path, nrows=100000)
else:
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)


# In[3]:


train.date = train.date.apply(lambda x: int(x.replace('-', '')))
test.date = test.date.apply(lambda x: int(x.replace('-', '')))


# In[4]:


train = train[train['date'] >= 20210101].reset_index(drop=True)


# In[5]:


def feature_engineer(train, test, split=20220501):
    train_len = len(train)
    data = pd.concat((train, test), sort=False).reset_index(drop=True)
    data = data.sort_values(by=['code', 'date'])
    
    stock_industry = pd.read_csv("stock_industry.csv", encoding="gbk")
    from sklearn.preprocessing import LabelEncoder
    lbe = LabelEncoder()
    stock_industry['industry'] = lbe.fit_transform(stock_industry['industry'])
    data = pd.merge(data, stock_industry[['code', 'industry']], how='left', on='code')

    # alpha net 
    length = 30
    for name in tqdm(['open', 'high', 'low', 'close', 'volume', 'amount', 'adjustflag', 'turn', 'pctChg', 'peTTM', 'psTTM', 'pcfNcfTTM', 'pbMRQ']):
#     for name in tqdm(['open']):
        roll_feature = []
        for i, group in data.groupby('code', sort=False)[name]:
            values = group.tolist()
            values = [0]*(length - 1) + values
            roll_feature = roll_feature + [values[i:i+length] for i in range(len(group))]
        data = pd.concat([data, pd.DataFrame(roll_feature, columns=[f'{name}_{i}' for i in range(length)])], axis=1).reset_index(drop=True)
    
    # generate label
    data['label'] = data.groupby('code').close.transform(lambda x:(x - x.shift(-14)) / (x + 1e-7) )
    data = data.dropna(subset = ['label'], inplace=False)
    data = data.replace(np.nan, 0)
    return data[data['date'] <= split].reset_index(drop=True), data[data['date'] > split].reset_index(drop=True)

train, test = feature_engineer(train, test)

ycol = 'label'
feature_names = list(
    filter(lambda x: x not in [ycol, 'code', 'date', ''], train.columns))

quantile_30, quantile_70 = train.label.quantile([0.3, 0.7]).values


def label_quantile(x):
    if x<quantile_30:
        return 0
    elif x<quantile_70:
        return 1
    else:
        return 2


# In[13]:


train.label = train.label.apply(label_quantile)
test.label = test.label.apply(label_quantile)


# In[16]:


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
                      eval_set=(X_val, Y_val),
                      plot=False,
                      verbose=500)


df_importance = pd.DataFrame({
    'column': feature_names,
    'importance': cat_model.feature_importances_,
})

# cat_model.save_model(f'cb_{frequency}.model')


# In[ ]:


print(df_importance)


# In[ ]:


cat_model.save_model(f'model/next_2week_alphanet30_1year.model')

