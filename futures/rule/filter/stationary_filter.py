import pandas as pd
import numpy as np
import re
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

def parse_condition(condition_str):
    """解析单个条件字符串"""
    operators = ['>=', '<=', '>', '<', '==', '!=']
    for op in operators:
        if op in condition_str:
            parts = condition_str.split(op)
            if len(parts) == 2:
                return (parts[0].strip(), op.strip(), float(parts[1].strip()))
    raise ValueError(f"无法解析条件: {condition_str}")

def generate_rule_returns(rule, data):
    """生成单个规则收益率序列（支持多品种）"""
    mask = pd.Series(True, index=data.index)
    conditions = rule['rule'].split(' AND ')
    
    for cond in conditions:
        feature, op, val = parse_condition(cond)
        col_data = data[feature]
        
        if op == '>=': mask &= (col_data >= val)
        elif op == '>': mask &= (col_data > val)
        elif op == '<=': mask &= (col_data <= val)
        elif op == '<': mask &= (col_data < val)
        elif op == '==': mask &= (col_data == val)
    
    # 应用信号过滤并保留品种信息
    signal_data = data[mask].copy()
    # 返回包含收益和品种的DataFrame，并调整收益方向
    returns_df = signal_data[['future_return', 'symbol']].copy()
    returns_df['future_return'] *= rule['direction']
    return returns_df

def evaluate_rule_stability(rule_returns, direction, years=[2020, 2021, 2022, 2023, 2024]):
    """多维度稳定性评估（包含品种数量检查）"""
    if not isinstance(rule_returns.index, pd.DatetimeIndex):
        raise ValueError("rule_returns的索引必须是DatetimeIndex")
    
    grouped = rule_returns.groupby(rule_returns.index.year)
    metrics = []
    year_details = {}
    action = "做多" if direction == 1 else "做空"
    
    for year, yearly_data in grouped:
        # if year not in years:
        #     continue
        # 检查交易天数
        if len(yearly_data) < 20:
            continue
        # 新增：检查品种数量
        num_symbols = yearly_data['symbol'].nunique()
        if num_symbols < 5:
            continue
        # 记录当年交易明细
        daily_trades = yearly_data.groupby(yearly_data.index.date).apply(
            lambda x: [f"{action}{sym}" for sym in x['symbol'].unique()]
        )
        for date, symbols in daily_trades.items():
            date_str = date.strftime("%Y-%m-%d")
            year_details[date_str] = symbols
        # 计算各项指标（使用正确的列名）
        returns_series = yearly_data['future_return']
        
        sharpe = returns_series.mean() / (returns_series.std() + 1e-8)
        win_rate = (returns_series > 0).mean()
        
        # 动态盈亏比
        gains = returns_series[returns_series > 0].sum()
        losses = abs(returns_series[returns_series < 0].sum())
        profit_ratio = gains / losses if losses > 0 else np.inf
        
        # 回撤分析
        cum_returns = (1 + returns_series).cumprod()
        peak = cum_returns.expanding().max()
        drawdown = (cum_returns - peak) / peak
        max_drawdown = drawdown.min()
        
        metrics.append({
            'year': year,
            'sharpe': sharpe,
            'win_rate': win_rate,
            'profit_ratio': profit_ratio,
            'max_drawdown': max_drawdown,
            'num_trades': len(yearly_data),
            'num_symbols': num_symbols  # 记录品种数量
        })
    
    # 要求至少有4个合格年份
    if len(metrics) < 4:
        return None, None
    formatted_details = {
        k: "\n".join(v) for k, v in year_details.items()
    }
    # 计算稳定性指标
    stability = {
        'sharpe_stable': all(m['sharpe'] > 1.0 for m in metrics),
        'win_stable': all(m['win_rate'] > 0.8 for m in metrics),
        'ratio_stable': all(m['profit_ratio'] > 1.2 for m in metrics),
        'drawdown_stable': all(m['max_drawdown'] > -0.2 for m in metrics),
        'avg_sharpe': np.mean([m['sharpe'] for m in metrics]),
        'avg_trades': np.mean([m['num_trades'] for m in metrics]),
        'avg_symbols': np.mean([m['num_symbols'] for m in metrics]),
        'year_details': formatted_details
    }
    
    if not (stability['sharpe_stable'] and stability['win_stable'] and stability['ratio_stable']):
        return None, None
    return metrics, stability

def process_batch(rules, price_data):
    """批量处理函数"""
    batch_results = []
    for _, rule in tqdm(rules.iterrows(), total=rules.shape[0], desc="Processing rules"):
        returns = generate_rule_returns(rule, price_data)
        metrics, stability = evaluate_rule_stability(returns, rule['direction'])
        if stability and stability['sharpe_stable']:
            batch_results.append({
                'rule': rule['rule'],
                'avg_sharpe': stability['avg_sharpe'],
                'avg_trades': stability['avg_trades'],
                **stability,
                'details': "\n".join(str(item) for item in metrics)
            })

    filtered_df = pd.DataFrame(batch_results)
    filtered_df.sort_values('avg_sharpe', ascending=False, inplace=True)
    filtered_df.to_csv("stable_rules.csv", encoding='utf-8-sig', index=False)
    return filtered_df

# 使用示例
if __name__ == "__main__":
    RULE_PATH = "../result/fine_rules.csv"
    DATA_PATH = "../data/feature_data/combined_data.csv"
    # 加载数据
    price_data = pd.read_csv(DATA_PATH)
    rules = pd.read_csv(RULE_PATH)
    
    # 执行筛选
    price_data['date'] = pd.to_datetime(price_data['date'])
    price_data.set_index('date', inplace=True)
    filtered_rules = process_batch(rules, price_data)
    
    print(f"筛选后保留规则数: {len(filtered_rules)}")
    print(filtered_rules[['rule', 'avg_sharpe', 'avg_trades']])