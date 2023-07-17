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
import talib as ta

# 获取全部股票的日K线数据
mkdir('stockdata/d_data')
raw_train_path = 'stockdata/d_train'
raw_test_path = 'stockdata/d_test'
train_path = 'stockdata/d_data/train.csv'
test_path = 'stockdata/d_data/test.csv'
industry_path = 'stockdata/stock_industry.csv'
mode = 'train'


# In[2]:


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


def tech_feature(data):
    print('begin technical indicator')
    # 移动平均线
    types = ['SMA','EMA','WMA','DEMA','TEMA', 'TRIMA','KAMA','MAMA','T3']
    timeperiods = [5, 30, 60, 120, 240]
    for i in range(len(types)):
        for d in timeperiods:
            data[f'{types[i]}_{d}'] = data.groupby('code', sort=False).close.transform(lambda x:ta.MA(x, timeperiod=d, matype=i))

    # 布林带 这里计算有重复
    data['H_line'] = data.groupby('code', sort=False).close.transform(lambda x:ta.BBANDS(x, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)[0])
    data['M_line'] = data.groupby('code', sort=False).close.transform(lambda x:ta.BBANDS(x, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)[1])
    data['L_line'] = data.groupby('code', sort=False).close.transform(lambda x:ta.BBANDS(x, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)[2])

    # 其他指标
    data['HT'] =  data.groupby('code', sort=False).close.transform(lambda x:ta.HT_TRENDLINE(x))
    data['MAVP'] = data.groupby('code', sort=False).close.transform(lambda x:ta.MAVP(x, periods=np.array([3]*len(x), dtype=float)))
    data['MIDPOINT'] = data.groupby('code', sort=False).close.transform(lambda x:ta.MIDPOINT(x))
    data['MIDPRICE'] = data.groupby('code', sort=False).apply(lambda x:ta.MIDPRICE(x.high, x.low)).reset_index(drop=True)
    data['SAR'] = data.groupby('code', sort=False).apply(lambda x:ta.SAR(x.high, x.low)).reset_index(drop=True)
    data['SAREXT'] = data.groupby('code', sort=False).apply(lambda x:ta.SAREXT(x.high, x.low)).reset_index(drop=True)
    return data

def feature_engineer(train, test, split=20220501):
    train_len = len(train)
    data = pd.concat((train, test), sort=False).reset_index(drop=True)
    data = data.sort_values(by=['code', 'date'])
    
    stock_industry = pd.read_csv(industry_path, encoding="gbk")
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
    
    # 技术指标
    data = tech_feature(data)
    
    # generate label
    data['label'] = data.groupby('code', sort=False).close.transform(lambda x:(x.shift(-14) - x) / (x + 1e-7) )
    data = data.dropna(subset = ['label'], inplace=False)
#     data = data.replace(np.nan, 0)
    return data[data['date'] <= split].reset_index(drop=True), data[data['date'] > split].reset_index(drop=True)


# In[6]:


train, test = feature_engineer(train, test)

# f_train_path = 'stockdata/d_data/f_train_debug.csv'
# f_test_path = 'stockdata/d_data/f_test_debug.csv'
# train.to_csv(f_train_path, index=False)
# test.to_csv(f_test_path, index=False)


# In[7]:


ycol = 'label'
feature_names = list(
    filter(lambda x: x not in [ycol, 'code', 'date', ''], train.columns))

# print(feature_names)


# In[8]:


quantile_30, quantile_70 = train.label.quantile([0.3, 0.7]).values
print('quantile_30:', quantile_30)
print('quantile_70:', quantile_70)


# In[9]:


def label_quantile(x):
    if x<quantile_30:
        return 0
    elif x<quantile_70:
        return 1
    else:
        return 2


# In[10]:


train.label = train.label.apply(label_quantile)
test.label = test.label.apply(label_quantile)


# In[11]:


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
                      verbose=500,
                      plot=False
                      )


df_importance = pd.DataFrame({
    'column': feature_names,
    'importance': cat_model.feature_importances_,
})

# cat_model.save_model(f'cb_{frequency}.model')


# In[12]:


pd.set_option('display.max_rows', None)
print(df_importance)


# In[13]:


cat_model.save_model(f'model/technical_indicators.model')

