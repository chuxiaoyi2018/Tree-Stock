import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import catboost as cb
import gc
from sklearn.preprocessing import LabelEncoder


def reduce_mem(df, cols):
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in tqdm(cols):
        col_type = df[col].dtypes
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024 ** 2
    print('{:.2f} Mb, {:.2f} Mb ({:.2f} %)'.format(start_mem, end_mem, 100 * (start_mem - end_mem) / start_mem))
    gc.collect()
    return df


raw_train_path = 'stockdata/d_train'
raw_test_path = 'stockdata/d_test'
# train_path = 'stockdata/d_data/train.csv'
# test_path = 'stockdata/d_data/test.csv'
mode = 'train'

train_path = 'stockdata/d_data/f_train.csv'
test_path = 'stockdata/d_data/f_test.csv'

if mode == 'debug':
    train = pd.read_csv(train_path, nrows=1000)
    test = pd.read_csv(test_path, nrows=100)
else:
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)


train['label'] = [0 if x>0 else 1 for x in (train.close - train.shift(-1).close)]
test['label'] = [0 if x>0 else 1 for x in (test.close - test.shift(-1).close)]

lbe = LabelEncoder()
train['code'] = lbe.fit_transform(train['code'])

lbe = LabelEncoder()
test['code'] = lbe.fit_transform(test['code'])

ycol = 'label'
feature_names = list(
    filter(lambda x: x not in [ycol, 'date', ''], train.columns))

# print(feature_names)
train, test = reduce_mem(train, list(train)), reduce_mem(test, list(test))


params = {'n_estimators':5000,
      'learning_rate': 0.01,
      'max_depth': 9,
      'early_stopping_rounds':1000,
      'loss_function':'Logloss',
      'max_bin':512,
      'subsample':0.8,
      'bootstrap_type':'Poisson',
      'random_seed':np.random.randint(0,2021)}

model = cb.CatBoostClassifier(eval_metric="AUC", task_type='GPU', **params)


lgb_model = model.fit(train[feature_names],
                      train[ycol],
                      # eval_names=['train', 'valid'],
                      # eval_set=[(X_train, Y_train), (X_val, Y_val)],
                      eval_set=(test[feature_names], test[ycol]),
                      verbose=200)


df_importance = pd.DataFrame({
    'column': feature_names,
    'importance': lgb_model.feature_importances_,
})

# lgb_model.save_model(f'cb_{frequency}.model')


lgb_model.save_model('cb_f1.model')

print(df_importance)