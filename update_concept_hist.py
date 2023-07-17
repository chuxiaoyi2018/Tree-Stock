# 这是一个获取A股板块历史交易记录并保存为.csv文件的Python代码，使用了akshare库来获取数据：
  
import akshare as ak
import pandas as pd
from tqdm import tqdm
from time import time

from utils import get_today_date


def update_concept_hist():
    # 设置起始日期和截止日期
    start_date, end_date = get_today_date()

    # 获取板块列表
    sector_df = ak.stock_board_concept_name_em()
    sector_list = sector_df['板块名称'].tolist()

    # 遍历每个板块
    begin_time = time()
    df_list = []
    for sector_name in sector_list:
        # sector_history_df = ak.stock_board_concept_hist_em(symbol=sector_name, start_date=start_date, end_date=end_date, adjust="")
        sector_history_df = ak.stock_board_concept_hist_em(symbol=sector_name)
        sector_history_df['板块名称'] = sector_name
        df_list.append(sector_history_df)
    df = pd.concat(df_list)
    df.to_csv(f'stockdata/concept_hist.csv', index=False)
    end_time = time()
    print(f'Time : {end_time - begin_time}')

update_concept_hist()