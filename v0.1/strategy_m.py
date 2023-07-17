from get_stock_data import Downloader, mkdir
from tqdm import tqdm
import os
import pandas as pd
import numpy as np
import catboost as cb
from sklearn.preprocessing import LabelEncoder

stockdata_path = './stockdata'

mode = "all"
train_path = './stockdata/m_train'
mkdir(train_path)
downloader = Downloader(train_path, 
                        date_start='1990-12-19', 
                        date_end='2022-04-22', 
                        frequency='m',
                        mode=mode)
downloader.run()

test_path = './stockdata/m_test'
mkdir(test_path)
downloader = Downloader(test_path, 
                        date_start='2022-04-23', 
                        date_end='2022-07-23', 
                        frequency='m',
                        mode=mode)
downloader.run()

def df_concat(csv_path):
    df_arr = []
    for path in tqdm(os.listdir(csv_path)):
        temp = pd.read_csv(csv_path + '/' + path, engine='python')
#         print(temp)
        if len(temp)==0:continue
        df_arr.append(temp)
    return pd.concat(df_arr)
    
    
train = df_concat(train_path)
test = df_concat(test_path)


def feature_engineer(train, test):
    train_len = len(train)
    data = pd.concat((train, test), sort=False).reset_index(drop=True)
    
    stock_industry = pd.read_csv("stock_industry.csv", encoding="gbk")
    from sklearn.preprocessing import LabelEncoder
    lbe = LabelEncoder()
    stock_industry['industry'] = lbe.fit_transform(stock_industry['industry'])
    data = pd.merge(data, stock_industry[['code', 'industry']], how='left', on='code')
    
    for name in tqdm(['close', 'volume', 'amount', 'turn', 'pctChg']):
        for day in ['7', '30']:
            rolling = data.groupby('code')[f'{name}'].rolling(window=int(day), center=False)
            data[f'{name}_rolling_{day}_mean'] = rolling.mean().reset_index(drop=True)
            data[f'{name}_rolling_{day}_max'] = rolling.max().reset_index(drop=True)
            data[f'{name}_rolling_{day}_min'] = rolling.min().reset_index(drop=True)
            data[f'{name}_rolling_{day}_sum'] = rolling.sum().reset_index(drop=True)
            data[f'{name}_rolling_{day}_median'] = rolling.median().reset_index(drop=True)
            data[f'{name}_rolling_{day}_skew'] = rolling.skew().reset_index(drop=True)
            data[f'{name}_rolling_{day}_kurt'] = rolling.kurt().reset_index(drop=True)
            data[f'{name}_rolling_{day}_std'] = rolling.std().reset_index(drop=True)
    return data.iloc[:train_len].reset_index(drop=True), data.iloc[train_len:].reset_index(drop=True)

train, test = feature_engineer(train, test)

lbe = LabelEncoder()
train['code'] = lbe.fit_transform(train['code'])

lbe = LabelEncoder()
test['code'] = lbe.fit_transform(test['code'])


train['label'] = [0 if x>0 else 1 for x in (train.close - train.shift(-1).close)]
test['label'] = [0 if x>0 else 1 for x in (test.close - test.shift(-1).close)]
train.to_csv(stockdata_path + '/' + 'm_train.csv')
test.to_csv(stockdata_path + '/' + 'm_test.csv')


ycol = 'label'
feature_names = list(
    filter(lambda x: x not in [ycol, 'date', ''], train.columns))


params = {'n_estimators':5000,
      'learning_rate': 0.05,
      'max_depth': 7,
      'early_stopping_rounds':1000,
      'loss_function':'Logloss',
      'max_bin':512,
      'subsample':0.8,
      'bootstrap_type':'Poisson',
      'random_seed':np.random.randint(0,2021)}

model = cb.CatBoostClassifier(eval_metric="AUC", task_type='GPU', **params)

X_train = train[feature_names]
Y_train = train[ycol]

X_val = test[feature_names]
Y_val = test[ycol]


lgb_model = model.fit(X_train,
                      Y_train,
                      # eval_names=['train', 'valid'],
                      # eval_set=[(X_train, Y_train), (X_val, Y_val)],
                      eval_set=(X_val, Y_val),
                      verbose=200)


df_importance = pd.DataFrame({
    'column': feature_names,
    'importance': lgb_model.feature_importances_,
})

# lgb_model.save_model(f'cb_{frequency}.model')

lgb_model.save_model(f'cb_month.model')

print(df_importance)