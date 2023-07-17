#!/usr/bin/env python
# coding: utf-8

# In[1]:

import sys
import os
import argparse
from tqdm import tqdm
import pandas as pd
import numpy as np
import catboost as cb
import datetime
import functools
from collections import defaultdict, Counter
import akshare as ak

from utils import reduce_mem, get_today_date, get_delta_ago_date, auc_multi_class, find_top_k, extract_digits


# 使用该股对应的板块的open close等，作为特征
def concept_feature(data):
    print('Begin concept feature')
    data['code_string'] = data.code
    data.code = data.code.apply(lambda x:int(x[3:]))
    data.date = data.date.apply(lambda x: int(x.replace('-', '')))
    concept_df = pd.read_csv(params.concept_path)[['代码', '板块名称']]
    concept_hist_df = pd.read_csv(params.concept_hist_path)
    concept_df['代码'] = concept_df['代码'].apply(lambda x:str(x).zfill(6))
    
    concept_hist_df['日期'] = concept_hist_df['日期'].apply(lambda x:int(x.replace('-', '')))
    
    concept_counter = Counter([c for c in concept_df['板块名称'].values if '昨日' not in c])

    concept_dic = defaultdict(list)
    for code, concept in concept_df.values:
        if '昨日' in concept:continue
        concept_dic[int(code)].append(concept)

    def compare_concept(x, y):
        x, y = concept_counter[x], concept_counter[y]
        if x < y:return -1
        if x > y: return 1
        return 0
    # 后期需要过滤掉所有的昨日！！，已经做了这个工作了
    for k, v in concept_dic.items():
        concept_dic[k] = sorted(v, key=functools.cmp_to_key(compare_concept))

    data['concept_0'] = data.code.apply(lambda x:concept_dic[int(x)][0] if len(concept_dic[x])>0 else np.nan)
    data['concept_1'] = data.code.apply(lambda x:concept_dic[int(x)][1] if len(concept_dic[x])>1 else np.nan)
    data['concept_2'] = data.code.apply(lambda x:concept_dic[int(x)][2] if len(concept_dic[x])>2 else np.nan)

    data['concept_-3'] = data.code.apply(lambda x:concept_dic[int(x)][-3] if len(concept_dic[x])>3 else np.nan)
    data['concept_-2'] = data.code.apply(lambda x:concept_dic[int(x)][-2] if len(concept_dic[x])>4 else np.nan)
    data['concept_-1'] = data.code.apply(lambda x:concept_dic[int(x)][-1] if len(concept_dic[x])>5 else np.nan)
    
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

def feature_engineer(data, window_size):
    data = data.sort_values(by=['code', 'date'])
    alpha_list = ['open', 'high', 'low', 'close', 'volume', 'amount', 'adjustflag', 'turn', 'pctChg', 'peTTM', 'psTTM', 'pcfNcfTTM', 'pbMRQ']
    
    stock_industry = pd.read_csv(params.industry_path, encoding="gbk")
#     stock_industry.code = stock_industry.code.apply(lambda x:int(x[3:]))
    
    from sklearn.preprocessing import LabelEncoder
    lbe = LabelEncoder()
    stock_industry['industry'] = lbe.fit_transform(stock_industry['industry'])
    data = pd.merge(data, stock_industry[['code', 'industry']], how='left', on='code')
    
    # concept feature
    data = concept_feature(data)
    alpha_list += [f'{x}{i}' for x in ['收盘', '换手率', '成交额'] for i in ['', '_1', '_2', '_-3', '_-2', '_-1']]
    data = reduce_mem(data, list(data))

    # alpha net
    length = window_size
    
    
    for name in tqdm(alpha_list):
        roll_feature = []
        for i, group in data.groupby('code', sort=False)[name]:
            values = group.tolist()
            values = [0]*(length - 1) + values
            roll_feature = roll_feature + [values[i:i+length] for i in range(len(group))]
        roll_columns = [f'{name}_dt{i}' for i in range(length)]
        data = pd.concat([data, pd.DataFrame(roll_feature, columns=roll_columns)], axis=1).reset_index(drop=True)
        data = reduce_mem(data, roll_columns)
    # generate label
    data['label'] = data.groupby('code', sort=False).close.transform(lambda x:(x.shift(-int(f"{params.window_size}")) - x) / (x + 1e-7) )
    train, test = data[data['label'].notna()].reset_index(drop=True), data[~data['label'].notna()].reset_index(drop=True)
    train = train.dropna(subset = ['label'], inplace=False)
    return train, test


# 这里还要做一个切分处理，在下载csv的时候，顺便按照时间顺序划分，这样的话之后只需要下载当天的数据酒可以了
def update_dataset():
    def df_concat(csv_path):
        df_arr = []
        header = ['date','code','open','high','low','close','preclose','volume','amount',
                 'adjustflag','turn','tradestatus','pctChg','peTTM','psTTM','pcfNcfTTM','pbMRQ','isST']
        for file_path in tqdm(os.listdir(csv_path)):
            try:
                if 'ipynb_checkpoints' in file_path:continue
                if file_path.split('.')[-1] != 'csv':continue
                temp = pd.read_csv(csv_path + '/' + file_path, engine='python')
                if 'date' not in temp.columns:
                    temp.columns = header
                temp['intDate'] = temp.date.apply(lambda x: int(x.replace('-', '')))
                temp = temp[temp['intDate'] >= int(params.date_abandon.replace('-', ''))].reset_index(drop=True)
                if len(temp)==0:continue
                df_arr.append(temp)                
            except pd.errors.EmptyDataError:
                print(file_path)
        return pd.concat(df_arr)
    train = df_concat(params.raw_data_path)
    train = train.drop('intDate', axis=1)
#     assert params.date_end in train.date.unique()
    train.to_csv(params.data_path, index=False)
    return

def preprocess():
    mode = params.mode
    if mode == 'debug':
        data = pd.read_csv(params.data_path)
        data = data[-100000:]
#         data = pd.read_csv(params.data_path, nrows=100000)
    else:
        data = pd.read_csv(params.data_path)
    # data['intDate'] = data.date.apply(lambda x: int(x.replace('-', '')))
    # data = data[data['intDate'] >= int(params.date_abandon.replace('-', ''))].reset_index(drop=True)
    # data = data.drop('intDate', axis=1)
    return data

def train_model(train, test):
    ycol = 'label'
    feature_names = list(
        filter(lambda x: x not in [ycol, 'code', 'code_string', 'date', ''] and '日期' not in x and '板块名称' not in x, train.columns))

    # print(feature_names)

    quantile_30, quantile_70 = train.label.quantile([0.3, 0.7]).values
    print('quantile_30:', quantile_30)
    print('quantile_70:', quantile_70)
    def label_quantile(x):
        if x<quantile_30:return 0
        elif x<quantile_70:return 1
        else:return 2

    train.label = train.label.apply(label_quantile)
    X_train = train[feature_names];Y_train = train[ycol]
    test.label = test.label.apply(label_quantile)
    X_val = test[feature_names];Y_val = test[ycol]
    

    model = cb.CatBoostClassifier(eval_metric="AUC", task_type='GPU', **params.model_params)
    cat_model = model.fit(X_train,
                          Y_train,
                          eval_set=(X_train[-10_0000:], Y_train[-10_0000:]), # 最好是用最近的看看
                          verbose=500)

    df_importance = pd.DataFrame({
        'column': feature_names,
        'importance': cat_model.feature_importances_,
    })

    # cat_model.save_model(f'cb_{frequency}.model')

    pd.set_option('display.max_rows', None)
    print(df_importance)
    cat_model.save_model(f'model/concept_{params.window_size}_{params.today_date}.model')

    # prediction and save
    prediction = cat_model.predict_proba(X_val)
    top_k_pred_probs, _, top_k_indices = find_top_k(prediction, test.label.values, k=params.topk)
    result = pd.DataFrame.from_dict({'股票代码':test.code_string[top_k_indices], 
                                     'date':test.date[top_k_indices], 
                                     'open':test.open[top_k_indices], 
                                     'high':test.high[top_k_indices], 
                                     'low':test.low[top_k_indices], 
                                     'close':test.close[top_k_indices], 
                                     'volume':test.volume[top_k_indices], 
                                     'amount': test.amount[top_k_indices],
                                     f'{params.window_size}天后下跌概率':top_k_pred_probs[:,0],
                                     f'{params.window_size}天后平盘概率':top_k_pred_probs[:,1], 
                                     f'{params.window_size}天后上涨概率':top_k_pred_probs[:,2]})
    latest_date = result.date.max()
    latest_result = result[result.date == result.date.max()].reset_index(drop=True)
    kcb_spot = ak.stock_zh_kcb_spot()
    cyb_spot = ak.stock_cy_a_spot_em()
    st_spot = ak.stock_zh_a_st_em()
    zt_spot = ak.stock_zt_pool_strong_em(date=latest_date)
    kcb_set = set(kcb_spot['代码'].apply(lambda x:extract_digits(x))).union(set(cyb_spot['代码'].apply(lambda x:extract_digits(x)))).union(set(st_spot['代码'].apply(lambda x:extract_digits(x)))).union(set(zt_spot['代码'].apply(lambda x:extract_digits(x))))

    latest_result = latest_result.drop([i for i,x in enumerate(latest_result['股票代码']) if x[3:] in kcb_set]).reset_index(drop=True)
    latest_result = latest_result[latest_result.open<1000].reset_index(drop=True)
    latest_result = latest_result[(latest_result.close - latest_result.open)/latest_result.open < 0.08].reset_index(drop=True)
    
    
    latest_result.to_csv(params.submit_path, index=False, encoding='utf_8_sig')

def main():
    
    # update dataset to today
    update_dataset()

    # preprocess ,,
    data = preprocess()

    # feature engineer
    train, test = feature_engineer(data, params.window_size)
    
    # train model
    train_model(train, test)

class params:
    mode = 'train'
    window_size = int(sys.argv[1])
    topk = 1000
    
    raw_data_path = 'stockdata/2023_by_date'
    data_path = 'stockdata/d_data/data.csv'
    test_path = 'stockdata/d_data/test.csv'
    industry_path = 'stockdata/stock_industry.csv'
    concept_path = 'stockdata/concept_df.csv'
    concept_hist_path = 'stockdata/concept_hist.csv'
    
    date_1year_ago, today_date = get_today_date()
    date_3day_ago, today_date = get_delta_ago_date(delta=3)
    
    date_abandon = date_1year_ago
    date_split = date_3day_ago
    date_end = sys.argv[2] # 不能选择今天的日期，周末的日期
    window_size_submit_path = f'submit/window_size{window_size}_{date_end}.csv'
    norepeat_window_size_submit_path = f'submit/norepat_window_size{window_size}_{date_end}.csv'
    submit_path = f'submit/{date_end}_window{window_size}.csv'
    
    model_params = {'n_estimators':1000,
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
    main()