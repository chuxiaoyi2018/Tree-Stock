import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OrdinalEncoder
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from collections import defaultdict
import os 
import gc
import time
import random
from sklearn.cluster import KMeans
from scipy import stats
import pickle
import re


from sklearn.metrics import roc_curve, auc


def extract_digits(s):
    """提取字符串中的数字"""
    return re.sub(r'\D', '', s)

def date2int(date):
    return int(date.replace('-',''))

def get_today_date():
    import datetime
    date_end = datetime.datetime.now().date()
    date_start = date_end - datetime.timedelta(days=365)
    date_start, date_end = date_start.strftime('%Y-%m-%d'), date_end.strftime('%Y-%m-%d')
    return date_start, date_end

def get_delta_ago_date(delta):
    import datetime
    date_end = datetime.datetime.now().date()
    date_start = date_end - datetime.timedelta(days=delta)
    date_start, date_end = date_start.strftime('%Y-%m-%d'), date_end.strftime('%Y-%m-%d')
    return date_start, date_end

def get_days(start, end):
    from datetime import datetime, timedelta
    """
    Return a list of days between start and end date.
    """
    days = []
    start_date = datetime.strptime(start, "%Y-%m-%d")
    end_date = datetime.strptime(end, "%Y-%m-%d")
    
    current_date = start_date
    while current_date <= end_date:
        days.append(current_date.strftime("%Y-%m-%d"))
        current_date += timedelta(days=1)
    return days

def auc_multi_class(y_pred, y_true):
    """
    计算多分类的AUC值
    :param y_pred: 预测的概率，二维数组，每一行代表一个样本的预测概率，每一列代表一个类别的预测概率
    :param y_true: 真实的标签，一维数组，每个元素代表一个样本的真实类别
    :return: AUC值
    """
    n_classes = y_pred.shape[1]
    total_auc = 0.0
    for i in range(n_classes):
        y_true_i = (y_true == i).astype(np.int32)
        y_pred_i = y_pred[:, i]
        fpr, tpr, _ = roc_curve(y_true_i, y_pred_i)
        auc_i = auc(fpr, tpr)
        total_auc += auc_i
        print(f'auc {i} : {auc_i}')
    return total_auc / n_classes

def find_top_k(pred_probs, true_labels, k):
    # pred_probs: 预测概率数组，形状为(num_samples, num_classes)
    # true_labels: 真实标签数组，形状为(num_samples,)
    # k: 需要找出的top k样本

    # 找出上涨概率最大的，根据最大概率排序，找出top k样本的索引
    top_k_indices = np.argsort(pred_probs[:,2])[::-1][:k]

    # 输出top k样本的预测概率数组和真实标签数组
    top_k_pred_probs = pred_probs[top_k_indices]
    top_k_true_labels = true_labels[top_k_indices]

    return top_k_pred_probs, top_k_true_labels, top_k_indices

# In[ ]:
def reduce_mem(df, cols):
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in cols:
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