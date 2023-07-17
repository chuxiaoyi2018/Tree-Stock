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
        if os.path.exists(f'stockdata/fund_flow/{stock_code}.csv'):
            fund_flow_df = pd.read_csv(f'stockdata/fund_flow/{stock_code}.csv')
            fund_flow_df['code'] = f"{stock_code}"
        else:
            market = stock_code[:2]
            stock = stock_code[3:]
            fund_flow_df = ak.stock_individual_fund_flow(stock=stock, market=market)
            fund_flow_df.to_csv(f'stockdata/fund_flow/{stock_code}.csv', index=False)
            sleep(1)
        df_list.append(fund_flow_df)
    df = pd.concat(df_list)
    df.to_csv(f'stockdata/fund_flow.csv', index=False)
    end_time = time()
    print(f'Time : {end_time - begin_time}')

update_individual_fund_flow_100()