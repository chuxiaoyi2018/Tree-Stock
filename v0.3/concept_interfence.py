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

from utils import reduce_mem, auc_multi_class, find_top_k


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
#     stock_industry.code = stock_industry.code.apply(lambda x:int(x[3:]))s
    
    from sklearn.preprocessing import LabelEncoder
    lbe = LabelEncoder()
    stock_industry['industry'] = lbe.fit_transform(stock_industry['industry'])
    data = pd.merge(data, stock_industry[['code', 'industry']], how='left', on='code')
    
    # concept feature
    data = concept_feature(data)
    alpha_list += [f'{x}{i}' for x in ['收盘', '换手率', '成交额'] for i in ['', '_1', '_2', '_-3', '_-2', '_-1']]
    #data = reduce_mem(data, list(data))

    # alpha net
    length = window_size
    
    
    for name in tqdm(alpha_list):
#     for name in tqdm(['open']):
        roll_feature = []
        for i, group in data.groupby('code', sort=False)[name]:
            values = group.tolist()
            values = [0]*(length - 1) + values
            roll_feature = roll_feature + [values[i:i+length] for i in range(len(group))]
        roll_columns = [f'{name}_dt{i}' for i in range(length)]
        data = pd.concat([data, pd.DataFrame(roll_feature, columns=roll_columns)], axis=1).reset_index(drop=True)
        #data = reduce_mem(data, roll_columns)
    
    # generate label
    data['label'] = data.groupby('code', sort=False).close.transform(lambda x:(x.shift(-14) - x) / (x + 1e-7) )
#     data = data.dropna(subset = ['label'], inplace=False)
#     data = data.replace(np.nan, 0)
    train, test = data[data['date'] <= params.date_split].reset_index(drop=True), data[data['date'] == params.date_split].reset_index(drop=True)
    train = train.dropna(subset = ['label'], inplace=False)
    return train, test

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

def preprocess():
    data = pd.read_csv(params.train_path)
    if params.mode == 'inference':
        from datetime import datetime, timedelta
        left_date = datetime.strptime(data['date'].max(), '%Y-%m-%d') - timedelta(days=30)
        left_date = left_date.strftime('%Y-%m-%d')
        data = data[data['date'] >= left_date].reset_index(drop=True)
#         data = data[:10000]
    else:
        raise ValueError('only support inference')
    return data

def infer_model(train, test):
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
    test.label = test.label.apply(label_quantile)
    X_train = train[feature_names];Y_train = train[ycol]
    X_val = test[feature_names];Y_val = test[ycol]

    model = cb.CatBoostClassifier(eval_metric="AUC", task_type='GPU', **params.model_params)
    model.load_model('model/concept_14.model')
    
    # prediction and save
#     import pdb;pdb.set_trace()
    prediction = model.predict_proba(X_val)
    auc_multi_class(prediction, test.label.values)
    top_k_pred_probs, top_k_true_labels, top_k_indices = find_top_k(prediction, test.label.values, k=params.topk)
    auc_multi_class(top_k_pred_probs, top_k_true_labels)
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
                                     f'{params.window_size}天后上涨概率':top_k_pred_probs[:,2], 
                                     '真实值':test.label.values[top_k_indices]})
    result.to_csv(params.submit_path, index=False)

def main():
    # 获取全部股票的日K线数据
    mkdir('stockdata/d_data')
    
    # update dataset to today
#     update_dataset()

    # preprocess 
    data = preprocess()

    # feature engineer
    train, test = feature_engineer(data, params.window_size)
    
    # train model
    infer_model(train, test)

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
    topk = 30 # 选择置信度排名前30的代码
    
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
    main()