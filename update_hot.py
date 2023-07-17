# 这是一个获取A股板块历史交易记录并保存为.csv文件的Python代码，使用了akshare库来获取数据：
  
import akshare as ak
import pandas as pd
from tqdm import tqdm
from time import time, sleep
import os

from utils import get_today_date
    
def update_individual_fund_flow_100():
    # 获取股票数据
    stock_df = pd.read_csv('stockdata/2023_by_date/2023-05-12.csv')

    # 遍历每个板块
    begin_time = time()
    df_list = []
    for stock_code in tqdm(stock_df['code']):
        market = stock_code[:2]
        stock = stock_code[3:]
        stock_hot_rank_detail_em_df = ak.stock_hot_rank_detail_em(symbol=f"{market.upper()}{stock}")
        df_list.append(stock_hot_rank_detail_em_df)
        sleep(1)
    df = pd.concat(df_list)
    df.to_csv(f'stockdata/hot/hot_rank_detail_em_df.csv', index=False)
    end_time = time()
    print(f'Time : {end_time - begin_time}')
    

update_individual_fund_flow_100()