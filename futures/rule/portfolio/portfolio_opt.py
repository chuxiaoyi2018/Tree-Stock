import pandas as pd
import numpy as np
from datetime import timedelta
from tqdm import tqdm
import re
from concurrent.futures import ProcessPoolExecutor


from pypfopt import discrete_allocation
from pypfopt import EfficientFrontier, risk_models, expected_returns, objective_functions

# ---------------------------- 收益率生成模块 ----------------------------
def parse_condition(condition_str):
    """解析单个条件字符串"""
    operators = ['>=', '<=', '>', '<', '==', '!=']
    for op in operators:
        if op in condition_str:
            parts = condition_str.split(op)
            if len(parts) == 2:
                return (parts[0].strip(), op, float(parts[1].strip()))
    raise ValueError(f"无法解析条件: {condition_str}")

def generate_rule_mask(rule_str, data):
    """生成规则触发掩码"""
    mask = pd.Series(True, index=data.index)
    for condition in rule_str.split(' AND '):
        try:
            feature, op, val = parse_condition(condition)
            col_data = data[feature]
            
            if op == '>=': mask &= (col_data >= val)
            elif op == '>': mask &= (col_data > val)
            elif op == '<=': mask &= (col_data <= val)
            elif op == '<': mask &= (col_data < val)
            elif op == '==': mask &= (col_data == val)
            elif op == '!=': mask &= (col_data != val)
        except KeyError:
            print(f"警告：特征列 {feature} 不存在，已跳过该条件")
            continue
    return mask

def calculate_rule_returns(rule_row, data, interval_days=14):
    """计算单个规则的收益率序列"""
    mask = generate_rule_mask(rule_row['rule'], data)
    trigger_dates = data[mask].index.unique().sort_values()
    
    # 应用间隔过滤
    valid_dates = []
    last_trigger = None
    for date in trigger_dates:
        if last_trigger is None or (date - last_trigger).days >= interval_days:
            valid_dates.append(date)
            last_trigger = date
    
    # 生成收益率序列
    returns = pd.Series(0.0, index=data.index, name=rule_row.name)
    for date in valid_dates:
        if date in data.index:
            returns[date] = data.loc[date, 'future_return'] * rule_row['direction']
    return returns

def generate_rule_returns(rule_path, data_path, year=2023, interval_days=14):
    """生成规则收益率矩阵"""
    # 加载数据
    data = pd.read_csv(data_path, 
                      parse_dates=['date'],
                      usecols=lambda c: c not in ['target', 'future_diff'])  # 排除无关列
    data = data[(data.date.dt.year == year) & ~data.future_return.isna()]
    data.set_index('date', inplace=True)
    
    # 加载规则
    rules = pd.read_csv(rule_path)
    rules = rules[['rule', 'direction']].reset_index()
    
    # 并行计算收益率
    returns_list = []
    with ProcessPoolExecutor() as executor:
        futures = []
        for _, row in rules.iterrows():
            futures.append(executor.submit(
                calculate_rule_returns,
                row,
                data,
                interval_days
            ))
        
        # 进度条监控
        for future in tqdm(futures, total=len(rules), desc="生成规则收益率"):
            returns_list.append(future.result())
    
    # 合并结果
    returns_df = pd.concat(returns_list, axis=1)
    returns_df.columns = [f'rule_{i+1}' for i in range(len(rules))]
    returns_df.to_csv("rule_returns_2023.csv", index=True)
    return returns_df

def max_sharpe():
    # 不使用eps，会报错，因为全零值太多
    eps = 0.000001
    returns = returns.replace([np.inf, -np.inf], np.nan)  
    returns = returns.fillna(eps)
    returns = returns.mask(np.abs(returns) < eps, eps)

    # 3. 准备优化参数
    mu = expected_returns.mean_historical_return(returns)
    S = risk_models.CovarianceShrinkage(returns).ledoit_wolf()
    
    # 4. 创建优化器
    ef = EfficientFrontier(mu, S, solver='SCS')
    
    # 5. 添加约束
    ef.add_objective(objective_functions.L2_reg, gamma=0.1)
    ef.add_constraint(lambda w: w <= 0.2)  # 单规则最大权重20%
    
    # 6. 执行优化
    weights = ef.max_sharpe()

def monte_carlo_optimization(returns, n_samples=10000):
    """蒙特卡洛组合优化（带稀疏性约束）"""
    np.random.seed(42)
    sharpe_max = -np.inf
    best_weights = None
    n_rules = len(returns.columns)
    
    for _ in tqdm(range(n_samples)):
        # 1. 随机选择5~10个规则
        k = np.random.randint(5, 11)
        selected = np.random.choice(n_rules, k, replace=False)
        
        # 2. 简化权重生成（单次生成）
        raw = np.random.rand(k)  # 生成初始随机数
        raw /= raw.sum()         # 初始归一化
        
        # 强制最大权重约束
        raw[raw.argmax()] = min(raw.max(), 0.2)  # 确保最大权重≤0.2
        adjusted = raw * (1 / raw.sum())          # 重新调整总和
        
        # 3. 构建最终权重
        full_weights = np.zeros(n_rules)
        full_weights[selected] = adjusted
        
        # 4. 计算夏普比率
        portfolio_returns = returns @ full_weights
        sharpe = portfolio_returns.mean() / (portfolio_returns.std() + 1e-8)
        
        # 更新最优组合
        if sharpe > sharpe_max:
            sharpe_max = sharpe
            best_weights = dict(zip(returns.columns, full_weights))

    return best_weights

# ---------------------------- 组合优化模块 ----------------------------
def portfolio_optimization(rule_path, returns_data):
    """执行组合优化"""
    # 1. 加载元数据
    rules_df = pd.read_csv(rule_path)
    
    # 2. 加载收益率数据
    returns = pd.read_csv(returns_data, index_col=0, parse_dates=True)

    # 数据清洗
    eps = 0.0
    returns = returns.replace([np.inf, -np.inf], np.nan)  
    returns = returns.fillna(eps)

    weights = monte_carlo_optimization(returns, n_samples=1000)
    
    # 离散化处理
    latest_prices = pd.Series(100, index=returns.columns).astype(float)  # 假设每个规则对应资产当前价格为100
    
    da = discrete_allocation.DiscreteAllocation(
        weights=weights,
        latest_prices=latest_prices,
        total_portfolio_value=100000  # 假设总投资额10万
    )
    
    # 分配结果
    alloc, leftover = da.greedy_portfolio()
    print(f"\n最终持仓分配:")
    for k, v in alloc.items():
        print(f"{k}: {v}手")
    print(f"剩余资金: {leftover:.2f}")
    
    # 7. 后处理权重
    weights = {'rule_1': 0.2, 'rule_2': 0.2, 'rule_3': 0.2, 'rule_4': 0.2, 'rule_5': 0.2, 'rule_6': 0.0, 'rule_7': 0.0, 'rule_8': 0.0, 'rule_9': 0.0, 'rule_10': 0.0, 'rule_11': 0.0, 'rule_12': 0.0, 'rule_13': 0.0, 'rule_14': 0.0, 'rule_15': 0.0, 'rule_16': 0.0, 'rule_17': 0.0, 'rule_18': 0.0, 'rule_19': 0.0, 'rule_20': 0.0, 'rule_21': 0.0, 'rule_22': 0.0, 'rule_23': 0.0, 'rule_24': 0.0, 'rule_25': 0.0, 'rule_26': 0.0, 'rule_27': 0.0, 'rule_28': 0.0, 'rule_29': 0.0, 'rule_30': 0.0, 'rule_31': 0.0, 'rule_32': 0.0, 'rule_33': 0.0, 'rule_34': 0.0, 'rule_35': 0.0, 'rule_36': 0.0, 'rule_37': 0.0, 'rule_38': 0.0, 'rule_39': 0.0, 'rule_40': 0.0, 'rule_41': 0.0, 'rule_42': 0.0, 'rule_43': 0.0, 'rule_44': 0.0, 'rule_45': 0.0, 'rule_46': 0.0, 'rule_47': 0.0, 'rule_48': 0.0, 'rule_49': 0.0, 'rule_50': 0.0, 'rule_51': 0.0, 'rule_52': 0.0, 'rule_53': 0.0, 'rule_54': 0.0, 'rule_55': 0.0, 'rule_56': 0.0, 'rule_57': 0.0, 'rule_58': 0.0, 'rule_59': 0.0, 'rule_60': 0.0, 'rule_61': 0.0, 'rule_62': 0.0, 'rule_63': 0.0, 'rule_64': 0.0, 'rule_65': 0.0, 'rule_66': 0.0, 'rule_67': 0.0, 'rule_68': 0.0, 'rule_69': 0.0, 'rule_70': 0.0, 'rule_71': 0.0, 'rule_72': 0.0}
    weights_series = pd.Series(weights).sort_values(ascending=False)
    selected_rules = weights_series.head(10)
    selected_rules /= selected_rules.sum()
    
    # 8. 输出结果
    print("\n优化后的规则组合：")
    for rule_id, weight in selected_rules.items():
        rule_idx = int(rule_id.split('_')[1]) - 1
        rule_data = rules_df.iloc[rule_idx]
        print(f"\n规则 {rule_id} (权重 {weight:.2%}):")
        print(f"条件: {rule_data['rule']}")
        print(f"方向: {'做空' if rule_data['direction'] == -1 else '做多'}")
        print(f"历史夏普: {rule_data['sharpe_ratio']:.2f} | 胜率: {rule_data['win_rate']:.2%}")
    
    print("\n=== 组合表现汇总 ===")
    # 计算组合整体表现
    portfolio_returns = returns[selected_rules.index].dot(selected_rules.values)
    sharpe = portfolio_returns.mean() / portfolio_returns.std()
    win_rate = (portfolio_returns > 0).mean()
    
    print(f"组合夏普比率: {sharpe:.2f}")
    print(f"组合胜率: {win_rate:.2%}")
    print(f"选中规则数量: {len(selected_rules)}")
    
    # 统计方向分布
    directions = rules_df.loc[[int(idx.split('_')[1])-1 for idx in selected_rules.index], 'direction']
    long_count = (directions == 1).sum()
    short_count = (directions == -1).sum()
    
    print(f"\n方向分布:")
    print(f"做多策略: {long_count}个 ({long_count/len(selected_rules):.0%})")
    print(f"做空策略: {short_count}个 ({short_count/len(selected_rules):.0%})")

    # 9. 保存结果
    pd.DataFrame({
        'rule_id': selected_rules.index,
        'weight': selected_rules.values
    }).to_csv('optimized_rules_weights.csv', index=False)

def backtest_portfolio(weights_path, data_path, year=2024, interval_days=14):
    """组合回测函数"""
    # 加载优化权重
    weights_df = pd.read_csv(weights_path)
    selected_rules = weights_df.set_index('rule_id')['weight']
    
    # 加载回测数据
    data = pd.read_csv(data_path, 
                      parse_dates=['date'],
                      usecols=lambda c: c not in ['target', 'future_diff'])
    data = data[data.date.dt.year == year]
    data.set_index('date', inplace=True)
    
    # 生成规则在回测期的收益率
    rule_returns = {}
    for rule_id in selected_rules.index:
        # 解析规则信息
        rule_num = int(rule_id.split('_')[1]) - 1
        rule_data = pd.read_csv(RULE_PATH).iloc[rule_num]
        
        # 生成信号
        mask = generate_rule_mask(rule_data['rule'], data)
        trigger_dates = data[mask].index.unique().sort_values()
        
        # 应用间隔过滤
        valid_dates = []
        last_trigger = None
        for date in trigger_dates:
            if last_trigger is None or (date - last_trigger).days >= interval_days:
                valid_dates.append(date)
                last_trigger = date
        
        # 生成收益率序列
        returns = pd.Series(0.0, index=data.index, name=rule_id)
        for date in valid_dates:
            if date in data.index:
                returns[date] = data.loc[date, 'future_return'] * rule_data['direction']
        
        rule_returns[rule_id] = returns
    
    # 组合收益率计算
    returns_df = pd.DataFrame(rule_returns)
    portfolio_returns = returns_df.dot(selected_rules)
    
    # 计算绩效指标
    sharpe = portfolio_returns.mean() / portfolio_returns.std()
    win_rate = (portfolio_returns > 0).mean()
    
    # 盈亏比计算
    gains = portfolio_returns[portfolio_returns > 0].mean()
    losses = portfolio_returns[portfolio_returns < 0].mean()
    profit_loss_ratio = abs(gains / losses) if losses != 0 else np.inf
    
    # 输出报告
    print("\n=== 回测报告 ===")
    print(f"回测周期: {year}年")
    print(f"夏普比率: {sharpe:.2f}")
    print(f"胜率: {win_rate:.2%}")
    print(f"盈亏比: {profit_loss_ratio:.2f}")
    print(f"总收益率: {portfolio_returns.sum():.2%}")
    
    # 可视化累计收益
    cumulative_returns = (1 + portfolio_returns).cumprod()
    cumulative_returns.plot(title="累计收益曲线")
    
    return {
        'sharpe': sharpe,
        'win_rate': win_rate,
        'profit_loss_ratio': profit_loss_ratio,
        'cumulative_returns': cumulative_returns
    }


# ---------------------------- 主执行流程 ----------------------------
# 这个方式只能说图一乐，直接废弃掉就行，没有实际意义！！！！！！！！
if __name__ == "__main__":
    # 参数配置
    RULE_PATH = "../result/backtest_metrics.csv"
    DATA_PATH = "../data/feature_data/combined_data.csv"
    INTERVAL_DAYS = 1
    
    # 步骤1：生成收益率数据
    print("正在生成规则收益率数据...")
    returns_df = generate_rule_returns(
        rule_path=RULE_PATH,
        data_path=DATA_PATH,
        year=2023,
        interval_days=INTERVAL_DAYS
    )
    
    # 步骤2：执行组合优化
    print("\n正在进行组合优化...")
    portfolio_optimization(
        rule_path=RULE_PATH,
        returns_data="rule_returns_2023.csv"
    )

    # 步骤3：执行回测
    print("\n正在进行回测...")
    backtest_results = backtest_portfolio(
        weights_path='optimized_rules_weights.csv',
        data_path=DATA_PATH.replace("2023", "2024"),  # 假设数据路径包含年份
        year=2024
    )