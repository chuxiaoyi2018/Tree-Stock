import pandas as pd
import numpy as np
import re
import os
from sklearn.linear_model import LogisticRegression
from joblib import dump

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def create_labels(df):
    """改进版多级标签生成"""
    try:
        # 创建分位数分箱
        neg_data = df[df['future_return'] < 0]
        pos_data = df[df['future_return'] >= 0]
        
        bins = []
        # 负收益部分
        if len(neg_data) >= 3:
            bins += [neg_data['future_return'].min(),
                    neg_data['future_return'].quantile(0.33),
                    neg_data['future_return'].quantile(0.66)]
        elif len(neg_data) > 0:
            bins.append(neg_data['future_return'].min())
        
        # 分界点
        bins.append(0)
        
        # 正收益部分
        if len(pos_data) >= 3:
            bins += [pos_data['future_return'].quantile(0.33),
                    pos_data['future_return'].quantile(0.66),
                    pos_data['future_return'].max()]
        elif len(pos_data) > 0:
            bins.append(pos_data['future_return'].max())
        
        # 去重排序
        bins = sorted(list(set(bins)))
        labels = []
        if len(bins) > 1:
            n_bins = len(bins)-1
            if n_bins == 6:
                labels = [-3, -2, -1, 1, 2, 3]
            else:
                labels = list(range(-n_bins//2, 0)) + list(range(1, n_bins//2+1))
            df['label'] = pd.cut(df['future_return'], bins=bins, labels=labels)
        
        return df.dropna(subset=['label']).astype({'label': int})
    
    except Exception as e:
        print(f"标签生成错误: {str(e)}")
        return pd.DataFrame()

def extract_rule_features(rule_str):
    """从单条规则中提取特征"""
    pattern = r"(\w+)[<>=]+"
    return list(set(re.findall(pattern, rule_str)))

def train_models():
    # 参数设置
    N_RULES = 24
    RULES_FILE = '../filter/stable_rules.csv'
    DATA_FILE = '../data/feature_data/combined_data.csv'
    SAVE_DIR = 'models'
    
    # 创建保存目录
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # 加载规则数据
    rules_df = pd.read_csv(RULES_FILE).head(N_RULES)
    
    # 加载交易数据
    full_df = pd.read_csv(DATA_FILE).dropna()
    full_df['date'] = pd.to_datetime(full_df['date'], errors='coerce')
    full_df = full_df[full_df['date'].dt.year <= 2022]
    labeled_df = create_labels(full_df)
    
    # 训练每个规则对应的模型
    used_features = set()
    for i, rule_row in rules_df.iterrows():
        try:
            # 提取当前规则特征
            rule_features = extract_rule_features(rule_row['rule'])
            
            current_features = set(rule_features)
            if len(current_features & used_features) > 1:
                print(f"跳过规则 {i}，重复特征: {current_features}")
                continue
            used_features = used_features | current_features
            print(f"训练模型{i}，使用特征：{rule_features}")

            # 数据准备
            X = labeled_df[rule_features]
            y = labeled_df['label']
            
            # 模型训练（移除multi_class参数）
            model = LogisticRegression(
                multi_class='multinomial',
                solver='saga',
                max_iter=200,  # 增加迭代次数保证收敛
                class_weight='balanced'
            )
            model.fit(X, y)
            
            # 保存模型
            dump({
                'model': model,
                'features': rule_features
            }, os.path.join(SAVE_DIR, f'lr_model_{i}.joblib'))
            
        except Exception as e:
            print(f"训练模型{i}失败: {str(e)}")
            continue

if __name__ == "__main__":
    train_models()