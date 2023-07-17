import pandas as pd
from tqdm import tqdm

df = pd.read_csv('stockdata/d_data/data.csv')

for i, group in tqdm(df.groupby('date')):
    group.to_csv(f'stockdata/2023_by_date/{i}.csv', index=False)