import pandas as pd
from tqdm import tqdm

df = pd.read_csv('stockdata/fund_flow.csv')

for i, group in tqdm(df.groupby('日期')):
    group.to_csv(f'stockdata/fund_flow_by_date/{i}.csv', index=False)