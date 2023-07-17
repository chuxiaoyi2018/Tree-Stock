# 这是一个获取A股板块历史交易记录并保存为.csv文件的Python代码，使用了akshare库来获取数据：
  
import akshare as ak
import pandas as pd
from tqdm import tqdm

# 设置起始日期和截止日期
start_date = '2022-01-01'
end_date = '2022-01-31'

# 获取板块列表
sector_df = ak.stock_board_industry_name_em()
sector_list = sector_df['板块名称'].tolist()

# 遍历每个板块
df_list = []
for sector_name in tqdm(sector_list):
    # 获取历史交易记录
    #sector_history_df = ak.stock_zh_a_hist(sector_name, start_date, end_date)
    sector_history_df = ak.stock_board_industry_hist_em(sector_name)
    # 添加板块名称列并保存为.csv文件
    sector_history_df['板块名称'] = sector_name
    df_list.append(sector_history_df)
    #sector_history_df.to_csv(f'{sector_name}.csv', index=False)
df = pd.concat(df_list)
df.to_csv(f'industry_hist.csv', index=False)
# 在这个程序中，首先调用`ak.stock_sector_spot()`方法获取A股板块列表。然后遍历每个板块，使用`ak.stock_zh_a_hist()`方法获取该板块从起始日期到截止日期的历史交易记录。最后，为DataFrame对象添加一个名为“板块名>称”的列，并将结果保存到以板块名称命名的.csv文件中。
