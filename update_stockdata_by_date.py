import baostock as bs
import pandas as pd
import os
from tqdm import tqdm
import datetime
from time import time
import sys

from utils import get_today_date


OUTPUT = './stockdata'


def mkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


class Downloader(object):
    def __init__(self,
                 output_dir,
                 mode="once",
                 frequency='d',
                 header=False,
                 date_start='1990-01-01',
                 date_end='2020-03-23'):
        self._bs = bs
        bs.login()
        self.frequency = frequency
        self.date_start = date_start
        self.header = header
        self.date_end = date_end
        self.output_dir = output_dir
        if self.frequency in ['5', '15', '30', '60']:
            self.fields = "date,time,code,open,high,low,close,volume,amount,adjustflag"
        elif self.frequency in ['d']:
            self.fields = "date,code,open,high,low,close,preclose,volume,amount,adjustflag,turn,tradestatus,pctChg,peTTM,psTTM,pcfNcfTTM,pbMRQ,isST"

        elif self.frequency in ['w', 'm']:
            self.fields = "date,code,open,high,low,close,volume,amount,adjustflag,turn,pctChg"
        self.mode = mode

    def exit(self):
        bs.logout()

    def get_codes_by_date(self, date):
        stock_rs = bs.query_all_stock(date)
        stock_df = stock_rs.get_data()
        return stock_df
    
    def run_once(self):
        stock_code = "bj.838810"
        df_code = bs.query_history_k_data_plus(stock_code, self.fields,
                                               start_date=self.date_start,
                                               end_date=self.date_end,
                                               frequency=self.frequency).get_data()
        df_code.to_csv(f'{self.output_dir}/stock_code.csv', index=False)
        
    def run_all(self):
        df_list = []
        stock_df = self.get_codes_by_date(self.date_end)
        # print(stock_df)
        # 说明可能是节假日
        if len(stock_df) != 0:
            # 5650
            for index, row in stock_df.iterrows():
                name = f'{row["code"]}.{row["code_name"]}'
                df_code = bs.query_history_k_data_plus(row["code"], 
                                                       self.fields, 
                                                       start_date=self.date_start, 
                                                       end_date=self.date_end, 
                                                       frequency=self.frequency).get_data()
                if len(df_code) == 0:continue
                df_list.append(df_code)
            df = pd.concat(df_list)
            df.to_csv(save_path, index=False)

    def run(self):
        header = ['date','code','open','high','low','close','preclose','volume','amount',
                 'adjustflag','turn','tradestatus','pctChg','peTTM','psTTM','pcfNcfTTM','pbMRQ','isST']
        if self.mode == "once":
            self.run_once()
        elif self.mode == "all":
            self.run_all()

if __name__ == '__main__':
    begin_time = time()
    
    mode = "all"
    root_path = 'stockdata/2023_by_date/'
    mkdir(root_path)
    _, date_end = get_today_date()
    date_start = date_end
    date_start, date_end = sys.argv[1], sys.argv[1]
    
    
    from utils import get_days
    for day in get_days(date_start, date_end):
        save_path = os.path.join(root_path, day + '.csv')
        downloader = Downloader(root_path, date_start=day, date_end=day, mode=mode)
        downloader.run()
    
    end_time = time()
    
    print(f'Time : {end_time - begin_time}')