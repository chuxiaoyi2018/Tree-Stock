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
import talib as ta
import functools
from collections import defaultdict, Counter

from utils import reduce_mem

def concept_feature(data):
    print('Begin concept feature')
    concept_df = pd.read_csv(path.concept_path)[['代码', '板块名称']]
    concept_hist_df = pd.read_csv(path.concept_hist_path)
    concept_df['代码'] = concept_df['代码'].apply(lambda x:str(x).zfill(6))
    
    concept_hist_df['日期'] = concept_hist_df['日期'].apply(lambda x:int(x.replace('-', '')))
    
    concept_counter = Counter([c for c in concept_df['板块名称'].values if '昨日' not in c])

    concept_dic = defaultdict(list)
    for code, concept in concept_df.values:
        if '昨日' in concept:continue
        concept_dic[code].append(concept)

    def compare_concept(x, y):
        x, y = concept_counter[x], concept_counter[y]
        if x < y:return -1
        if x > y: return 1
        return 0
    for k, v in concept_dic.items():
        concept_dic[k] = sorted(v, key=functools.cmp_to_key(compare_concept))

    data['concept_0'] = data.code.apply(lambda x:concept_dic[x][0] if len(concept_dic[x])>0 else np.nan)
    data['concept_1'] = data.code.apply(lambda x:concept_dic[x][1] if len(concept_dic[x])>1 else np.nan)
    data['concept_2'] = data.code.apply(lambda x:concept_dic[x][2] if len(concept_dic[x])>2 else np.nan)

    data['concept_-3'] = data.code.apply(lambda x:concept_dic[x][-3] if len(concept_dic[x])>3 else np.nan)
    data['concept_-2'] = data.code.apply(lambda x:concept_dic[x][-2] if len(concept_dic[x])>4 else np.nan)
    data['concept_-1'] = data.code.apply(lambda x:concept_dic[x][-1] if len(concept_dic[x])>5 else np.nan)
    
    data = pd.merge(data, concept_hist_df, left_on=['date', 'concept_0'], right_on=['日期', '板块名称'], how='left')
    data = pd.merge(data, concept_hist_df, left_on=['date', 'concept_1'], right_on=['日期', '板块名称'], how='left', suffixes=(None, '_1'))
    data = pd.merge(data, concept_hist_df, left_on=['date', 'concept_2'], right_on=['日期', '板块名称'], how='left', suffixes=(None, '_2'))
    data = pd.merge(data, concept_hist_df, left_on=['date', 'concept_-3'], right_on=['日期', '板块名称'], how='left', suffixes=(None, '_-3'))
    data = pd.merge(data, concept_hist_df, left_on=['date', 'concept_-2'], right_on=['日期', '板块名称'], how='left', suffixes=(None, '_-2'))
    data = pd.merge(data, concept_hist_df, left_on=['date', 'concept_-1'], right_on=['日期', '板块名称'], how='left', suffixes=(None, '_-1'))
    
    
    # label encoder
    concept_labelencoder = {c:i for i, c in enumerate(np.unique(concept_df['板块名称'].values))}
    concept_labelencoder.update({np.nan:np.nan})
    data['concept_0'] = data['concept_0'].apply(lambda x:concept_labelencoder[x])
    data['concept_1'] = data['concept_1'].apply(lambda x:concept_labelencoder[x])
    data['concept_2'] = data['concept_2'].apply(lambda x:concept_labelencoder[x])
    
    data['concept_-3'] = data['concept_-3'].apply(lambda x:concept_labelencoder[x])
    data['concept_-2'] = data['concept_-2'].apply(lambda x:concept_labelencoder[x])
    data['concept_-1'] = data['concept_-1'].apply(lambda x:concept_labelencoder[x])
    return data

def feature_engineer(train, test, window_size, split=20220501):
    train_len = len(train)
    data = pd.concat((train, test), sort=False).reset_index(drop=True)
    data = data.sort_values(by=['code', 'date'])
    
    stock_industry = pd.read_csv(path.industry_path, encoding="gbk")
    from sklearn.preprocessing import LabelEncoder
    lbe = LabelEncoder()
    stock_industry['industry'] = lbe.fit_transform(stock_industry['industry'])
    data = pd.merge(data, stock_industry[['code', 'industry']], how='left', on='code')
    
    # concept feature
    data = concept_feature(data)
    data = reduce_mem(data, list(data))

    # alpha net
    length = window_size
    alpha_list = ['open', 'high', 'low', 'close', 'volume', 'amount', 'adjustflag', 'turn', 'pctChg', 'peTTM', 'psTTM', 'pcfNcfTTM', 'pbMRQ']
    alpha_list += [f'{x}{i}' for x in ['收盘', '换手率', '成交额'] for i in ['', '_1', '_2', '_-3', '_-2', '_-1']]
    for name in tqdm(alpha_list):
#     for name in tqdm(['open']):
        roll_feature = []
        for i, group in data.groupby('code', sort=False)[name]:
            values = group.tolist()
            values = [0]*(length - 1) + values
            roll_feature = roll_feature + [values[i:i+length] for i in range(len(group))]
        roll_columns = [f'{name}_dt{i}' for i in range(length)]
        data = pd.concat([data, pd.DataFrame(roll_feature, columns=roll_columns)], axis=1).reset_index(drop=True)
        data = reduce_mem(data, roll_columns)
    
    # generate label
    data['label'] = data.groupby('code', sort=False).close.transform(lambda x:(x.shift(-14) - x) / (x + 1e-7) )
    data = data.dropna(subset = ['label'], inplace=False)
#     data = data.replace(np.nan, 0)
    return data[data['date'] <= split].reset_index(drop=True), data[data['date'] > split].reset_index(drop=True)


# In[3]:


def main(args):
    # 获取全部股票的日K线数据
    mkdir('stockdata/d_data')
    mode = args.mode

    if mode == 'debug':
        train = pd.read_csv(path.train_path, nrows=100000)
        test = pd.read_csv(path.test_path, nrows=100000)
    else:
        train = pd.read_csv(path.train_path)
        test = pd.read_csv(path.test_path)


    train.date = train.date.apply(lambda x: int(x.replace('-', '')))
    test.date = test.date.apply(lambda x: int(x.replace('-', '')))

    train.code = train.code.apply(lambda x:x[3:])
    test.code = test.code.apply(lambda x:x[3:])

    train = train[train['date'] >= 20210101].reset_index(drop=True)

    train, test = feature_engineer(train, test, args.window_size)

    ycol = 'label'
    feature_names = list(
        filter(lambda x: x not in [ycol, 'code', 'date', ''] and '日期' not in x and '板块名称' not in x, train.columns))

    # print(feature_names)

    quantile_30, quantile_70 = train.label.quantile([0.3, 0.7]).values
    print('quantile_30:', quantile_30)
    print('quantile_70:', quantile_70)
    def label_quantile(x):
        if x<quantile_30:
            return 0
        elif x<quantile_70:
            return 1
        else:
            return 2

    train.label = train.label.apply(label_quantile)
    test.label = test.label.apply(label_quantile)


    params = {'n_estimators':5000,
          'learning_rate': 0.05,
          'max_depth': 7,
          'early_stopping_rounds':1000,
          'loss_function':'MultiClass',
           'classes_count':3,
          'max_bin':512,
          'subsample':0.8,
          'bootstrap_type':'Poisson',
          'random_seed':np.random.randint(0,2021)}

    model = cb.CatBoostClassifier(eval_metric="AUC", task_type='GPU', **params)

    X_train = train[feature_names]
    Y_train = train[ycol]

    X_val = test[feature_names]
    Y_val = test[ycol]


    cat_model = model.fit(X_train,
                          Y_train,
                          eval_set=(X_val, Y_val),
                          verbose=500)


    df_importance = pd.DataFrame({
        'column': feature_names,
        'importance': cat_model.feature_importances_,
    })

    # cat_model.save_model(f'cb_{frequency}.model')

    pd.set_option('display.max_rows', None)
    print(df_importance)

    cat_model.save_model(f'model/concept_{args.window_size}.model')

class path:
    raw_train_path = 'stockdata/d_train'
    raw_test_path = 'stockdata/d_test'
    train_path = 'stockdata/d_data/train.csv'
    test_path = 'stockdata/d_data/test.csv'
    industry_path = 'stockdata/stock_industry.csv'
    concept_path = 'stockdata/concept_df.csv'
    concept_hist_path = 'stockdata/concept_hist_df.csv'

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='debug', type=str)
    parser.add_argument('--window_size', type=int)
    args = parser.parse_args()
    main(args)




