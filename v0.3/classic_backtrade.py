#!/usr/bin/env python
# coding: utf-8

# In[1]:

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

from utils import reduce_mem, get_today_date, get_delta_ago_date, auc_multi_class, find_top_k, get_days, date2int, extract_digits


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
    if params.mode != 'debug':
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
        if params.mode != 'debug':
            data = reduce_mem(data, roll_columns)
    # generate label
    data['future_close'] = data.groupby('code', sort=False).close.transform(lambda x:(x.shift(-int(f"{params.window_size}")) ))
    data['label'] = data.groupby('code', sort=False).close.transform(lambda x:(x.shift(-int(f"{params.window_size}")) - x) / (x + 1e-7) )
    return data


# 这里还要做一个切分处理，在下载csv的时候，顺便按照时间顺序划分，这样的话之后只需要下载当天的数据酒可以了
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

def update_dataset():
    train = df_concat(params.raw_data_path)
    train = train.drop('intDate', axis=1)
    train.to_csv(params.data_path, index=False)
    return

def preprocess(kcb_set):
    mode = params.mode
    if mode == 'debug':
        data = pd.read_csv(params.data_path)
        data = data[-100000:].reset_index(drop=True)
    else:
        data = pd.read_csv(params.data_path)
    data = data.drop([i for i,x in enumerate(data['code']) if str(x[3:]) in kcb_set]).reset_index(drop=True)
    return data

def train_model(data, day, backtrade_path):
    ycol = 'label'
    
    train = data[data.date < int(day.replace('-',''))].reset_index(drop=False)
    test = data[data.date == int(day.replace('-',''))].reset_index(drop=False)
    if 'index' in train.columns:train, test = train.drop(columns=['index']), test.drop(columns=['index'])
    
    feature_names = list(
        filter(lambda x: x not in [ycol, 'index', 'code', 'code_string', 'date', '', 'future_close'] and '日期' not in x and '板块名称' not in x, train.columns))
    
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

    pd.set_option('display.max_rows', None)
    print(df_importance)
    cat_model.save_model(f'model/concept_{params.window_size}_{params.today_date}.model')

    # prediction and save
    prediction = cat_model.predict_proba(X_val)
    top_k_pred_probs, _, top_k_indices = find_top_k(prediction, test.label.values, k=params.topk)
    
    up_or_down = ['涨' if x > 0 else '跌' for x in (test.future_close[top_k_indices] - test.close[top_k_indices])]
    result = pd.DataFrame.from_dict({'股票代码':test.code_string[top_k_indices], 
                                     'date':test.date[top_k_indices], 
                                     'volume':test.volume[top_k_indices], 
                                     'amount': test.amount[top_k_indices],
                                     'open':test.open[top_k_indices], 
                                     'high':test.high[top_k_indices], 
                                     'low':test.low[top_k_indices], 
                                     'close':test.close[top_k_indices], 
                                     f'{params.window_size}天后股价':test.future_close[top_k_indices],
                                     f'{params.window_size}天后下跌概率':top_k_pred_probs[:,0],
                                     f'{params.window_size}天后平盘概率':top_k_pred_probs[:,1], 
                                     f'{params.window_size}天后上涨概率':top_k_pred_probs[:,2],
                                     f'{params.window_size}天后是否涨跌':up_or_down})
    latest_result = result[result.date == result.date.max()].reset_index(drop=True)
    # norepeat_result = result.drop_duplicates(['股票代码'])
    
    kcb_spot = ak.stock_zh_kcb_spot()
    cyb_spot = ak.stock_cy_a_spot_em()
    st_spot = ak.stock_zh_a_st_em()
    kcb_set = set(kcb_spot['代码'].apply(lambda x:extract_digits(x))).union(set(cyb_spot['代码'].apply(lambda x:extract_digits(x)))).union(set(st_spot['代码'].apply(lambda x:extract_digits(x))))

    latest_result = latest_result.drop([i for i,x in enumerate(latest_result['股票代码']) if x[3:] in kcb_set]).reset_index(drop=True)
    latest_result = latest_result[latest_result.open<500].reset_index(drop=True)
    latest_result.to_csv(backtrade_path, index=False, encoding='utf_8_sig')
    
def get_kcb_set():
    kcb_spot = ak.stock_zh_kcb_spot()
    cyb_spot = ak.stock_cy_a_spot_em()
    st_spot = ak.stock_zh_a_st_em()
    kcb_set = set(kcb_spot['代码'].apply(lambda x:extract_digits(x))).union(set(cyb_spot['代码'].apply(lambda x:extract_digits(x)))).union(set(st_spot['代码'].apply(lambda x:extract_digits(x))))
    return kcb_set
    
def compute_profit(backtrade_path, kcb_set, base_shares=100, selected_topk=10):
    df = pd.read_csv(backtrade_path)
    count = 0
    profit = 0
    cost = 0
    win_count = 0
    
    df = df.drop([i for i,x in enumerate(df['股票代码']) if x[3:] in kcb_set]).reset_index(drop=True)
    for i, row in df.iterrows():
        future_price = row[f'{params.window_size}天后股价']
        if (row.close - row.open)/row.open >= 0.09:continue
        if (future_price - row.open)/row.open >= 0.20:continue
        if row.close > 200:continue
        future_price = row[f'{params.window_size}天后股价']
        profit += base_shares * (future_price - row.close)
        win_count += (future_price - row.close) > 0
        print(f"【日期】{row.date} : 【股票代码】{row['股票代码']} : 【利润】{round(base_shares * (future_price - row.close), 2)} : 【收盘价】{row.close} : 【{params.window_size}天后股价】{future_price}")
        cost += base_shares * row.close
        count += 1
        if count == selected_topk:break
        
    print(f'买前{count}的股票，盈亏是：{round(profit, 2)}' )
    print(f'买前{count}的股票，开销是：{cost}' )
    print(f'买前{count}的股票，胜率是：{win_count/count}' )
    return profit, cost, win_count, count

def train_and_backtrade(start_day, end_day, kcb_set):
    # update dataset to today
    update_dataset()

    # preprocess 
    data = preprocess(kcb_set)

    # feature engineer
    data = feature_engineer(data, params.window_size)

    # train model
    for day in get_days(start_day, end_day):
        if date2int(day) not in data.date.values:continue
        backtrade_path = f'backtrade/classic_backtrade_{day}_{params.window_size}.csv'
        train_model(data, day, backtrade_path)

    # compute profit
    total_profit = 0
    total_cost = 0
    total_win_count = 0
    total_count = 0
    for day in get_days(start_day, end_day):
        if date2int(day) not in data.date.values:continue
        backtrade_path = f'backtrade/classic_backtrade_{day}_{params.window_size}.csv'
        print(f'-----------------------{day}-----------------------')
        profit, cost, win_count, count = compute_profit(backtrade_path, kcb_set, selected_topk=params.selected_topk)
        total_profit += profit
        total_cost += cost
        total_win_count += win_count
        total_count += count
    return total_profit, total_cost, total_win_count, total_count

def backtrade(start_day, end_day, kcb_set):
    total_profit = 0
    total_cost = 0
    total_win_count = 0
    total_count = 0
    for day in get_days(start_day, end_day):
        backtrade_path = f'backtrade/classic_backtrade_{day}_{params.window_size}.csv'
        if not os.path.exists(backtrade_path):continue
        print(f'-----------------------{day}-----------------------')
        profit, cost, win_count, count = compute_profit(backtrade_path, kcb_set, selected_topk=params.selected_topk)
        total_profit += profit
        total_cost += cost
        total_win_count += win_count
        total_count += count
    return total_profit, total_cost, total_win_count, total_count
    

def main():
    start_day = '2023-01-05'
    end_day = '2023-04-10'
    kcb_set = get_kcb_set()
    
    if params.mode == 'train_and_backtrade' or params.mode == 'debug':
        total_profit, total_cost, total_win_count, total_count = train_and_backtrade(start_day, end_day, kcb_set)
    elif params.mode == 'backtrade':
        total_profit, total_cost, total_win_count, total_count = backtrade(start_day, end_day, kcb_set)
    else:
        raise ValueError
    print(f'买前{params.selected_topk}的股票，{start_day}到{end_day}，总盈亏是：{total_profit}' )
    print(f'买前{params.selected_topk}的股票，{start_day}到{end_day}，总开销是：{total_cost}' )
    print(f'买前{params.selected_topk}的股票，{start_day}到{end_day}，盈亏/开销比是：{total_profit/total_cost}' )
    print(f'买前{params.selected_topk}的股票，{start_day}到{end_day}，总胜率是：{total_win_count/total_count}' )
            
class params:
    mode = 'train_and_backtrade'
    window_size = 3
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
    #date_start = '2022-06-02'
    date_end = today_date # 不能选择今天的日期，周末的日期
    
    # backtrade
    # base_shares = 100
    selected_topk = 3
    
    
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
    

def print_attributes():
    attributes = dir(params)
    attributes = [attr for attr in attributes if not attr.startswith('__') and not attr.endswith('__')]
    for attr in attributes:
        print(f"{attr} : {getattr(params, attr)}")
    
    
if __name__ == '__main__':
    main()
    print_attributes()