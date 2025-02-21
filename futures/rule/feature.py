import akshare as ak
import pandas as pd
from collections import Counter

def get_combined_position(date, variety, N=6):
    """获取并处理持仓数据"""
    # 获取各交易所数据(示例仅用DCE)
    dce_data = ak.futures_dce_position_rank(date=date)
    
    # 数据清洗和预处理
    combined = []
    for symbol, df in dce_data.items():
        # 转换数值格式
        numeric_cols = ['long_open_interest', 'short_open_interest', 'vol']
        for col in numeric_cols:
            df[col] = df[col].astype(str).str.replace(',', '').astype(float)
        
        # 添加品种字段
        df['variety'] = df['variety'].str.strip()
        combined.append(df)
    
    all_data = pd.concat(combined)
    variety_data = all_data[all_data['variety'] == variety]
    
    # 计算会员排名
    vol_rank = variety_data.groupby('vol_party_name')['vol'].sum().nlargest(20).index.tolist()
    long_rank = variety_data.groupby('long_party_name')['long_open_interest'].sum().nlargest(20).index.tolist()
    short_rank = variety_data.groupby('short_party_name')['short_open_interest'].sum().nlargest(20).index.tolist()
    
    # 合并排名并选取头部会员
    merged = list(set(vol_rank + long_rank + short_rank))
    counter = Counter(vol_rank + long_rank + short_rank)
    top_members = [x[0] for x in counter.most_common(N)]
    
    return variety_data, top_members

def calculate_spider_features(date, variety):
    """计算蜘蛛网策略特征"""
    # 获取基础数据
    data, top_members = get_combined_position(date, variety)
    
    # 划分头部会员和其他会员
    top_data = data[data['vol_party_name'].isin(top_members)]
    other_data = data[~data['vol_party_name'].isin(top_members)]
    
    # 计算市场总体指标
    total_oi = (data['long_open_interest'].sum() + data['short_open_interest'].sum()) / 2
    total_vol = data['vol'].sum()
    stat_threshold = total_oi / total_vol if total_vol != 0 else 0
    
    # 计算会员特征
    def calc_group_features(group):
        group = group.groupby('vol_party_name').agg({
            'long_open_interest': 'sum',
            'short_open_interest': 'sum',
            'vol': 'sum'
        }).reset_index()
        group['stat'] = (group['long_open_interest'] + group['short_open_interest']) / group['vol']
        return group
    
    top_features = calc_group_features(top_data)
    other_features = calc_group_features(other_data)
    
    # 区分知情/非知情投资者
    informed = top_features[top_features['stat'] > stat_threshold]
    uninformed = pd.concat([top_features[top_features['stat'] <= stat_threshold], other_features])
    
    # 计算情绪指标
    def calc_sentiment(df):
        long = df['long_open_interest'].sum()
        short = df['short_open_interest'].sum()
        return (long - short) / (long + short) if (long + short) > 0 else 0
    
    ITS = calc_sentiment(informed)
    UTS = calc_sentiment(uninformed)
    MSD = ITS - UTS
    
    return {
        'date': date,
        'variety': variety,
        'ITS': ITS,
        'UTS': UTS,
        'MSD': MSD,
        'stat_threshold': stat_threshold
    }

# 使用示例
if __name__ == "__main__":
    features = calculate_spider_features('20231115', 'JM')
    print("蜘蛛网策略特征:")
    print(f"知情投资者情绪(ITS): {features['ITS']:.4f}")
    print(f"非知情投资者情绪(UTS): {features['UTS']:.4f}") 
    print(f"市场情绪差异(MSD): {features['MSD']:.4f}")
    print(f"知情判断阈值: {features['stat_threshold']:.2f}")
