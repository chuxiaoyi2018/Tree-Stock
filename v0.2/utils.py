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

import torch
import torch.nn as nn
import torch.nn.functional as F

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

class StockDNN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size = [1024, 512, 256]) -> None:
        super(StockDNN, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.loss_func = nn.CrossEntropyLoss()
        
        self.mlp = nn.Sequential(
            nn.Linear(self.input_dim, hidden_size[0]),
#             nn.Dropout(0.5),
#             nn.BatchNorm1d(hidden_size[0]),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size[0], hidden_size[1]),
#             nn.Dropout(0.5),
#             nn.BatchNorm1d(hidden_size[1]),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size[1], hidden_size[2]),
#             nn.Dropout(0.5),
#             nn.BatchNorm1d(hidden_size[2]),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size[2], self.output_dim),
        )
    
    def forward(self, x):
        output = self.mlp(x)
        return output
            
    def compute_loss(self, x, target):
        output = self.forward(x)
        loss = self.loss_func(output, target.long())
        return loss