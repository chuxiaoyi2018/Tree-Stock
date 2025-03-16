import pandas as pd
import numpy as np
import re
from tqdm import tqdm

def parse_conditions(rule_str):
    """解析规则字符串为条件列表"""
    pattern = r'([\w_]+)\s*([<>]=?|=)\s*([\d\.eE+-]+)'
    matches = re.findall(pattern, rule_str)
    conditions = []
    for feature, op, value in matches:
        # 标准化操作符
        if op == '=>': op = '>='
        elif op == '=<': op = '<='
        try: value = float(value)
        except: continue
        conditions.append((feature, op, value))
    return conditions

def apply_rules_to_data(rules_df, test_data):
    """应用所有规则到测试数据"""
    long_rule_masks = {}
    short_rule_masks = {}
    for idx, row in tqdm(rules_df.iterrows(), total=len(rules_df), desc="Processing Rules"):
        conditions = parse_conditions(row['rule'])
        if not conditions:
            continue

        mask = pd.Series(True, index=test_data.index)
        for feature, op, value in conditions:
            if feature not in test_data.columns:
                mask &= False
                continue

            if op == '>=':
                cond = test_data[feature] >= value
            elif op == '<=':
                cond = test_data[feature] <= value
            elif op == '>':
                cond = test_data[feature] > value
            elif op == '<':
                cond = test_data[feature] < value
            elif op == '==':
                cond = test_data[feature] == value
            else:
                continue

            mask &= cond

        if row['direction'] == 1:
            long_rule_masks[f'long_rule_{idx}'] = mask.astype(int)
        elif row['direction'] == -1:
            short_rule_masks[f'short_rule_{idx}'] = mask.astype(int)

    # 一次性连接所有规则列
    long_rule_masks_df = pd.DataFrame(long_rule_masks)
    short_rule_masks_df = pd.DataFrame(short_rule_masks)
    test_data = pd.concat([test_data, long_rule_masks_df, short_rule_masks_df], axis=1)

    return test_data

def calculate_triggers(test_data, rules_df):
    """计算触发规则"""
    long_rule_columns = [f'long_rule_{idx}' for idx in rules_df[rules_df['direction'] == 1].index]
    short_rule_columns = [f'short_rule_{idx}' for idx in rules_df[rules_df['direction'] == -1].index]

    # 计算触发规则列表
    test_data['long_trigger'] = test_data[long_rule_columns].apply(
        lambda row: [f'long_rule_{i}' for i, val in enumerate(row) if val == 1], axis=1)
    test_data['short_trigger'] = test_data[short_rule_columns].apply(
        lambda row: [f'short_rule_{i}' for i, val in enumerate(row) if val == 1], axis=1)

    # 计算触发数量
    test_data['long_trigger_num'] = test_data[long_rule_columns].sum(axis=1)
    test_data['short_trigger_num'] = test_data[short_rule_columns].sum(axis=1)

    # 计算触发权重
    long_rule_sharpe = rules_df[rules_df['direction'] == 1]['sharpe_ratio'].values
    short_rule_sharpe = rules_df[rules_df['direction'] == -1]['sharpe_ratio'].values
    test_data['long_trigger_weight'] = (test_data[long_rule_columns] * long_rule_sharpe).sum(axis=1)
    test_data['short_trigger_weight'] = (test_data[short_rule_columns] * short_rule_sharpe).sum(axis=1)

    return test_data

def calculate_sliding_window(test_data):
    """计算7日滑动窗口，引入衰减因子"""
    test_data = test_data.sort_values(['symbol', 'date'])

    def weighted_rolling_sum(x):
        weights = np.array([decay_factor ** i for i in range(len(x))][::-1])
        return (x * weights).sum()


    # 计算每个symbol的滑动窗口总和
    test_data['long_window_trigger_num'] = test_data.groupby('symbol')['long_trigger_num'].transform(
        lambda x: x.rolling(window_size, min_periods=1).sum())
    test_data['short_window_trigger_num'] = test_data.groupby('symbol')['short_trigger_num'].transform(
        lambda x: x.rolling(window_size, min_periods=1).sum())

    # 计算触发权重的滑动窗口，引入衰减
    test_data['long_window_trigger_weight'] = test_data.groupby('symbol')['long_trigger_weight'].transform(
        lambda x: x.rolling(window_size, min_periods=1).apply(weighted_rolling_sum))
    test_data['short_window_trigger_weight'] = test_data.groupby('symbol')['short_trigger_weight'].transform(
        lambda x: x.rolling(window_size, min_periods=1).apply(weighted_rolling_sum))

    # 过滤触发次数小于阈值的行
    test_data = test_data[
        (test_data['long_window_trigger_weight'] >= trigger_threshold) |
        (test_data['short_window_trigger_weight'] >= trigger_threshold)
    ]

    return test_data

def calculate_metrics(data):
    """计算各类指标"""
    # 多头指标计算
    long_data = data[data['long_window_trigger_weight'] >= trigger_threshold].copy()
    long_data['future_return'] = long_data['future_return'] * 1  # 多头方向为1
    long_avg_return = long_data['future_return'].mean()
    long_sharpe_ratio = long_avg_return / long_data['future_return'].std() if long_data['future_return'].std() != 0 else np.nan
    long_win_rate = (long_data['future_return'].dropna() > 0).mean()
    long_min_return = long_data['future_return'].min()

    long_profitable_trades = long_data[long_data['future_return'] > 0]['future_return']
    long_losing_trades = long_data[long_data['future_return'] < 0]['future_return']
    long_avg_profit = long_profitable_trades.mean() if not long_profitable_trades.empty else 0
    long_avg_loss = -long_losing_trades.mean() if not long_losing_trades.empty else 0
    long_profit_loss_ratio = long_avg_profit / long_avg_loss if long_avg_loss != 0 else np.nan

    # 空头指标计算
    short_data = data[data['short_window_trigger_weight'] >= trigger_threshold].copy()
    short_data['future_return'] = short_data['future_return'] * -1  # 空头方向为 -1
    short_avg_return = short_data['future_return'].mean()
    short_sharpe_ratio = short_avg_return / short_data['future_return'].std() if short_data['future_return'].std() != 0 else np.nan
    short_win_rate = (short_data['future_return'].dropna() > 0).mean()
    short_min_return = short_data['future_return'].min()

    short_profitable_trades = short_data[short_data['future_return'] > 0]['future_return']
    short_losing_trades = short_data[short_data['future_return'] < 0].dropna()['future_return']
    short_avg_profit = short_profitable_trades.mean() if not short_profitable_trades.empty else 0
    short_avg_loss = -short_losing_trades.mean() if not short_losing_trades.empty else 0
    short_profit_loss_ratio = short_avg_profit / short_avg_loss if short_avg_loss != 0 else np.nan

    print("多头指标:")
    print(f"平均收益率 (Avg Return): {long_avg_return}")
    print(f"夏普比率 (Sharpe Ratio): {long_sharpe_ratio}")
    print(f"胜率 (Win Rate): {long_win_rate}")
    print(f"最小收益率 (Min Return): {long_min_return}")
    print(f"盈亏比 (Profit Loss Ratio): {long_profit_loss_ratio}")

    print("\n空头指标:")
    print(f"平均收益率 (Avg Return): {short_avg_return}")
    print(f"夏普比率 (Sharpe Ratio): {short_sharpe_ratio}")
    print(f"胜率 (Win Rate): {short_win_rate}")
    print(f"最小收益率 (Min Return): {short_min_return}")
    print(f"盈亏比 (Profit Loss Ratio): {short_profit_loss_ratio}")

if __name__ == "__main__":
    # 配置参数
    combined_data_csv = './data/feature_data/combined_data.csv'
    rule_set_csv = './result/backtest_metrics.csv'
    aggregate_signals_csv = './daily/aggregate_signals.csv'

    # Configuration
    window_size = 1
    decay_factor = 0  # 衰减因子
    backtest_start_date = "2024-01-01"
    backtest_end_date = "2025-03-30"
    trigger_threshold = 1  # 触发次数阈值
    
    # 加载数据
    combined_data = pd.read_csv(combined_data_csv)
    combined_data['date'] = pd.to_datetime(combined_data['date'])
    
    # 过滤测试数据
    test_data = combined_data[
        (combined_data['date'] >= pd.Timestamp(backtest_start_date)) &
        (combined_data['date'] <= pd.Timestamp(backtest_end_date))
    ].copy()

    # 加载规则
    rules_df = pd.read_csv(rule_set_csv).iloc[[0,1,2,3,4]]

    # 应用规则到测试数据
    test_data = apply_rules_to_data(rules_df, test_data)
    
    # 计算触发规则
    test_data = calculate_triggers(test_data, rules_df)
    
    # 计算滑动窗口
    test_data = calculate_sliding_window(test_data)
    
   # 计算指标
    calculate_metrics(test_data)

    # 保存结果
    columns_to_keep = ['date', 'symbol', 'future_return', 'long_trigger', 'short_trigger', 'long_trigger_num', 'short_trigger_num', 'long_window_trigger_num', 'short_window_trigger_num', 'long_trigger_weight', 'short_trigger_weight', 'long_window_trigger_weight', 'short_window_trigger_weight']
    test_data = test_data[columns_to_keep]
    test_data.to_csv(aggregate_signals_csv, encoding='utf-8-sig', index=False)

    print(f"回测结果已保存至 {aggregate_signals_csv}")
    print(f"总记录数: {len(test_data):,}")
