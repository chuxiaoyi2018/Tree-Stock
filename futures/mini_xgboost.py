import akshare as ak
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
import joblib
from tqdm import tqdm
import re

# 配置全局参数
pd.set_option('display.float_format', lambda x: '%.3f' % x)

# 获取期货代码列表
def get_futures_symbols():
    futures_list = ak.futures_hist_table_em()  # 获取所有期货的现货信息
    return futures_list['合约中文代码'].unique().tolist()

# 获取期货历史数据
def get_futures_data(symbol, raw_data_dir):
    os.makedirs(raw_data_dir, exist_ok=True)
    file_path = os.path.join(raw_data_dir, f'{symbol}.csv')

    column_mapping = {
        '时间': 'date',
        '开盘': 'open',
        '最高': 'high',
        '最低': 'low',
        '收盘': 'close',
        '涨跌': 'change',
        '涨跌幅': 'change_percentage',
        '成交量': 'volume',
        '成交额': 'turnover',
        '持仓量': 'open_interest'
    }

    if os.path.exists(file_path):
        try:
            df = pd.read_csv(file_path)
            df.rename(columns=column_mapping, inplace=True)
            df['date'] = pd.to_datetime(df['date'])
            return df
        except Exception as e:
            print(f"Error reading data from {file_path}: {e}")
    else:
        try:
            df = ak.futures_hist_em(symbol=symbol, period="daily")  # 获取日线数据
            df.rename(columns=column_mapping, inplace=True)
            df['date'] = pd.to_datetime(df['date'])
            df['symbol'] = symbol
            df.to_csv(file_path, index=False)
            return df
        except Exception as e:
            print(f"Error occurred while fetching data for symbol {symbol}: {e}")
            return None

# 特征工程
def feature_engineering(df):
    # 基本价格特征
    df['return'] = df['close'].pct_change()  # 收益率
    df['volatility'] = df['close'].rolling(window=5).std()  # 5 日波动率
    df['momentum'] = df['close'] / df['close'].shift(5) - 1  # 5 日动量
    df['ma_5'] = df['close'].rolling(window=5).mean()  # 5 日均线
    df['ma_10'] = df['close'].rolling(window=10).mean()  # 10 日均线
    df['ma_20'] = df['close'].rolling(window=20).mean()  # 20 日均线
    df['ma_30'] = df['close'].rolling(window=30).mean()  # 30 日均线
    df['ma_diff_5_10'] = df['ma_5'] - df['ma_10']  # 5 日与 10 日均线差
    df['ma_diff_5_20'] = df['ma_5'] - df['ma_20']  # 5 日与 20 日均线差
    df['ma_diff_5_30'] = df['ma_5'] - df['ma_30']  # 5 日与 30 日均线差
    df['ma_diff_10_20'] = df['ma_10'] - df['ma_20']  # 10 日与 20 日均线差
    df['ma_diff_10_30'] = df['ma_10'] - df['ma_30']  # 10 日与 30 日均线差
    df['ma_diff_20_30'] = df['ma_20'] - df['ma_30']  # 20 日与 30 日均线差

    # 不同时间窗口的价格统计特征
    df['max_7'] = df['close'].rolling(window=7).max()  # 7 日最高价
    df['min_7'] = df['close'].rolling(window=7).min()  # 7 日最低价
    df['max_30'] = df['close'].rolling(window=30).max()  # 30 日最高价
    df['min_30'] = df['close'].rolling(window=30).min()  # 30 日最低价
    df['range_7'] = df['max_7'] - df['min_7']  # 7 日价格范围
    df['range_30'] = df['max_30'] - df['min_30']  # 30 日价格范围

    # 量价特征
    df['volume_return'] = df['volume'].pct_change()  # 成交量收益率
    df['price_volume_ratio'] = df['close'] / df['volume']  # 价量比
    df['volume_ma_5'] = df['volume'].rolling(window=5).mean()  # 5 日成交量均线
    df['volume_ma_10'] = df['volume'].rolling(window=10).mean()  # 10 日成交量均线
    df['volume_ma_diff_5_10'] = df['volume_ma_5'] - df['volume_ma_10']  # 5 日与 10 日成交量均线差

    # 相对强弱指标（RSI）
    delta = df['close'].diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    avg_gain = up.rolling(window=14).mean()
    avg_loss = down.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # 偏度因子
    df['skew_7'] = df['close'].rolling(window=7).skew()
    df['skew_30'] = df['close'].rolling(window=30).skew()

    # 峰度因子
    df['kurt_7'] = df['close'].rolling(window=7).kurt()
    df['kurt_30'] = df['close'].rolling(window=30).kurt()

    # 波动因子
    df['volatility_10'] = df['close'].rolling(window=10).std()
    df['volatility_20'] = df['close'].rolling(window=20).std()
    df['volatility_30'] = df['close'].rolling(window=30).std()
    df['volatility_ratio_5_10'] = df['volatility'] / df['volatility_10']
    df['volatility_ratio_5_20'] = df['volatility'] / df['volatility_20']
    df['volatility_ratio_5_30'] = df['volatility'] / df['volatility_30']

    # 流动性因子
    df['turnover_rate'] = df['volume'] / (df['total_shares'] if 'total_shares' in df.columns else 1)
    df['liquidity_5'] = df['turnover_rate'].rolling(window=5).mean()
    df['liquidity_10'] = df['turnover_rate'].rolling(window=10).mean()
    df['liquidity_20'] = df['turnover_rate'].rolling(window=20).mean()
    df['liquidity_30'] = df['turnover_rate'].rolling(window=30).mean()

    # 均价突破因子
    df['avg_price_5'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
    df['avg_price_ma_5'] = df['avg_price_5'].rolling(window=5).mean()
    df['avg_price_ma_10'] = df['avg_price_5'].rolling(window=10).mean()
    df['avg_price_ma_20'] = df['avg_price_5'].rolling(window=20).mean()
    df['avg_price_ma_30'] = df['avg_price_5'].rolling(window=30).mean()
    df['avg_price_break_5'] = (df['close'] > df['avg_price_ma_5']).astype(int)
    df['avg_price_break_10'] = (df['close'] > df['avg_price_ma_10']).astype(int)
    df['avg_price_break_20'] = (df['close'] > df['avg_price_ma_20']).astype(int)
    df['avg_price_break_30'] = (df['close'] > df['avg_price_ma_30']).astype(int)

    # 资金流向因子
    df['money_flow'] = df['volume'] * df['close']
    df['money_flow_ma_5'] = df['money_flow'].rolling(window=5).mean()
    df['money_flow_ma_10'] = df['money_flow'].rolling(window=10).mean()
    df['money_flow_ma_20'] = df['money_flow'].rolling(window=20).mean()
    df['money_flow_ma_30'] = df['money_flow'].rolling(window=30).mean()

    # 乖离率因子
    df['bias_5'] = (df['close'] - df['ma_5']) / df['ma_5']
    df['bias_10'] = (df['close'] - df['ma_10']) / df['ma_10']
    df['bias_20'] = (df['close'] - df['ma_20']) / df['ma_20']
    df['bias_30'] = (df['close'] - df['ma_30']) / df['ma_30']

    # 二分类目标，涨为 1，跌为 0
    df['target'] = (df['close'].shift(-target_day) > df['close']).astype(int)

    # 计算未来 N 天后的收益率
    df['future_diff'] = df['close'].shift(-target_day) - df['close']
    df['future_return'] = df['future_diff'] / df['close']

    # 去除包含缺失值的行
    df = df.dropna()

    # 处理无穷大或数值过大的值
    columns_to_normalize = [col for col in df.columns if col not in non_normalize_columns]
    # 使用 .loc 方法避免 SettingWithCopyWarning
    df.loc[:, columns_to_normalize] = df[columns_to_normalize].replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=columns_to_normalize)

    # 归一化处理
    columns_to_normalize = [col for col in df.columns if col not in non_normalize_columns]
    scaler = MinMaxScaler()
    if len(df) > 0:
        df.loc[:, columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])
    return df

def train_model(df):
    features = [col for col in df.columns if col not in excluded_columns]
    if os.path.exists(model_path):
        print("Loading existing model...")
        model = joblib.load(model_path)
    else:
        X = df[features]
        y = df['target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        # 将 eval_metric 参数移到初始化中
        model = XGBClassifier(n_estimators=2000, max_depth=5, learning_rate=0.05, random_state=42, eval_metric="error", early_stopping_rounds=200)
        # 定义验证集
        eval_set = [(X_test, y_test)]
        # 训练模型，传入 eval_set 和 early_stopping_rounds 并开启 verbose
        model.fit(X_train, y_train, eval_set=eval_set, verbose=True)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model Accuracy: {accuracy:.2f}")
        # 调用打印特征重要性的函数
        print_feature_importance(model, features)
        # 保存模型
        os.makedirs('./model', exist_ok=True)
        joblib.dump(model, model_path)
        print(f"Model saved to {model_path}")

    export_split_rules_to_csv(model, features, './model/split_rules.csv')
    return model

def print_feature_importance(model, feature_names):
    """
    计算并打印 XGBoost 模型的特征重要性
    :param model: 训练好的 XGBoost 模型
    :param feature_names: 特征名称列表
    """
    # 计算特征重要性
    feature_importance_weight = model.get_booster().get_score(importance_type='weight')
    feature_importance_gain = model.get_booster().get_score(importance_type='gain')
    feature_importance_cover = model.get_booster().get_score(importance_type='cover')
    # 确保所有特征都在特征重要性字典中，缺失的特征重要性设为 0
    for feature in feature_names:
        if feature not in feature_importance_weight:
            feature_importance_weight[feature] = 0
        if feature not in feature_importance_gain:
            feature_importance_gain[feature] = 0
        if feature not in feature_importance_cover:
            feature_importance_cover[feature] = 0
    # 将特征重要性转换为 DataFrame 以便查看
    df_weight = pd.DataFrame.from_dict(feature_importance_weight, orient='index', columns=['weight'])
    df_gain = pd.DataFrame.from_dict(feature_importance_gain, orient='index', columns=['gain'])
    df_cover = pd.DataFrame.from_dict(feature_importance_cover, orient='index', columns=['cover'])
    # 确保特征名称一致
    df_weight = df_weight.reindex(feature_names)
    df_gain = df_gain.reindex(feature_names)
    df_cover = df_cover.reindex(feature_names)
    # 对每个 DataFrame 按重要性降序排序并取前 50 个
    df_weight_sorted = df_weight.sort_values(by='weight', ascending=False).head(50)
    df_gain_sorted = df_gain.sort_values(by='gain', ascending=False).head(50)
    df_cover_sorted = df_cover.sort_values(by='cover', ascending=False).head(50)
    print("\n基于 weight 的前 50 个特征重要性:")
    print(df_weight_sorted)
    print("\n基于 gain 的前 50 个特征重要性:")
    print(df_gain_sorted)
    print("\n基于 cover 的前 50 个特征重要性:")
    print(df_cover_sorted)
    # 创建保存目录
    save_dir = './feature_importance/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # 保存排序后的结果到本地
    df_weight_sorted.to_csv(os.path.join(save_dir, 'feature_importance_weight_top50.csv'))
    df_gain_sorted.to_csv(os.path.join(save_dir, 'feature_importance_gain_top50.csv'))
    df_cover_sorted.to_csv(os.path.join(save_dir, 'feature_importance_cover_top50.csv'))

# 回测函数
def backtest(df, model, filename='backtest_results.png'):
    features = [col for col in df.columns if col not in excluded_columns]
    df['prediction'] = model.predict(df[features])
    df['strategy_return'] = df['prediction'] * df['return']  # 策略收益
    df['cumulative_strategy_return'] = (1 + df['strategy_return']).cumprod()  # 策略累计收益
    df['cumulative_market_return'] = (1 + df['return']).cumprod()  # 市场累计收益

    # 绘制回测结果
    plt.figure(figsize=(12, 6))
    plt.plot(df['date'], df['cumulative_strategy_return'], label='Strategy Return')
    plt.plot(df['date'], df['cumulative_market_return'], label='Market Return')
    plt.legend()
    plt.title('Backtest Results')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')

    # 保存图像到本地
    plt.savefig(filename)  # 保存为指定的文件名
    plt.close()  # 关闭当前图形，释放内存
    print(f"Backtest results saved to {filename}")

def export_split_rules_to_csv(model, feature_names, output_path='leaf_rules.csv'):
    """
    导出完整的决策路径到 CSV，包含规则序列、累计增益、覆盖度和叶子节点预测值
    :param model: 训练好的 XGBoost 模型
    :param feature_names: 特征名称列表
    :param output_path: 输出文件路径
    """
    trees = model.get_booster().get_dump(with_stats=True, dump_format="text")
    all_paths = []

    # 解析单棵树的模式
    split_pattern = re.compile(
        r'^(\d+):\[(\w+)\s*([<>]=?)\s*([\d\.eE+-]+)\]\s*'  # 节点ID和分裂条件
        r'yes=(\d+),\s*no=(\d+).*?'                        # 分支指向
        r'gain=([\d\.eE+-]+),\s*cover=([\d\.eE+-]+)'       # 增益和覆盖度
    )
    
    leaf_pattern = re.compile(
        r'^(\d+):leaf=([\d\.eE+-]+)(,\s*cover=([\d\.eE+-]+))?'  # 叶子节点
    )

    for tree in trees:
        # 构建节点结构
        nodes = {}
        for line in tree.split('\n'):
            line = line.strip()
            if not line:
                continue
            
            # 解析分裂节点
            split_match = split_pattern.match(line)
            if split_match:
                node_id = int(split_match.group(1))
                nodes[node_id] = {
                    'type': 'split',
                    'feature': split_match.group(2),
                    'op': split_match.group(3),
                    'threshold': split_match.group(4),
                    'yes': int(split_match.group(5)),
                    'no': int(split_match.group(6)),
                    'gain': float(split_match.group(7)),
                    'cover': float(split_match.group(8))
                }
                continue
                
            # 解析叶子节点
            leaf_match = leaf_pattern.match(line)
            if leaf_match:
                node_id = int(leaf_match.group(1))
                nodes[node_id] = {
                    'type': 'leaf',
                    'value': float(leaf_match.group(2)),
                    'cover': float(leaf_match.group(4)) if leaf_match.group(4) else 0.0
                }
        
        # DFS 遍历收集路径
        def dfs(node_id, path, current_gain, current_cover):
            node = nodes.get(node_id)
            if not node:
                return
            
            if node['type'] == 'leaf':
                # 计算预测概率
                pred = 1 / (1 + np.exp(-node['value']))  # sigmoid转换
                all_paths.append({
                    'path': path.copy(),
                    'total_gain': current_gain,
                    'total_cover': current_cover + node['cover'],  # 累加叶子节点cover
                    'yes_prob': pred,
                    'no_prob': 1 - pred
                })
                return
            
            # 添加当前分裂规则
            new_rule = f"{node['feature']} {node['op']} {node['threshold']}"
            new_path = path + [new_rule]
            
            # 递归遍历分支
            dfs(node['yes'], new_path, current_gain + node['gain'], current_cover + node['cover'])
            dfs(node['no'], new_path, current_gain + node['gain'], current_cover + node['cover'])
        
        # 从根节点（通常为0）开始遍历
        dfs(0, [], 0.0, 0.0)

    # 转换为 DataFrame
    max_depth = max(len(p['path']) for p in all_paths)
    columns = [f'rule_{i}' for i in range(max_depth)] + ['total_gain', 'total_cover', 'yes_prob', 'no_prob']
    
    data = []
    for path in all_paths:
        row = {f'rule_{i}': rule for i, rule in enumerate(path['path'])}
        row.update({
            'total_gain': path['total_gain'],
            'total_cover': path['total_cover'],
            'yes_prob': path['yes_prob'],
            'no_prob': path['no_prob']
        })
        data.append(row)
    
    df = pd.DataFrame(data).fillna('')
    df = df.sort_values('total_gain', ascending=False)
    df.to_csv(output_path, index=False)
    print(f"规则已保存至 {output_path}")

def compute_sharpe_by_rules(model, data, split_rules_csv, output_csv='sharpe_by_rules.csv'):
    """
    根据决策规则计算每个规则在各个symbol下的最大收益率、最小收益率、平均收益率以及夏普比率
    并保存结果为CSV文件，按夏普比率降序排序
    :param model: XGBoost模型
    :param data: 包含特征和target的DataFrame
    :param split_rules_csv: 存储分裂规则的CSV文件路径
    :param output_csv: 输出的CSV文件路径
    """
    # 读取分裂规则
    rules_df = pd.read_csv(split_rules_csv)
    # 初始化结果列表
    results = []

    # 遍历每一条规则路径
    for idx, row in rules_df.iterrows():
        # 提取规则
        rules = [row[col] for col in rules_df.columns if 'rule_' in col and pd.notna(row[col]) and row[col] != '']
        if not rules:
            continue  # 如果没有规则，跳过

        # 构建布尔遮罩
        mask = pd.Series([True] * len(data))
        for rule in rules:
            # 解析规则，假设规则格式为 "feature operator threshold"
            parts = rule.split()
            if len(parts) != 3:
                continue  # 规则格式不正确，跳过
            feature, operator, threshold = parts
            try:
                threshold = float(threshold)
            except ValueError:
                continue  # 阈值不是数值，跳过

            if operator == '>=':
                mask &= (data[feature] >= threshold)
            elif operator == '>':
                mask &= (data[feature] > threshold)
            elif operator == '<=':
                mask &= (data[feature] <= threshold)
            elif operator == '<':
                mask &= (data[feature] < threshold)
            elif operator == '==':
                mask &= (data[feature] == threshold)
            else:
                continue  # 未支持的操作符，跳过

        # 根据遮罩筛选数据
        filtered_data = data[mask]
        if filtered_data.empty:
            continue  # 无匹配数据，跳过

        # 按symbol分组计算指标
        grouped = filtered_data.groupby('symbol')
        for symbol, group in grouped:
            if group.empty:
                continue
            max_return = group['future_return'].max()
            min_return = group['future_return'].min()
            avg_return = group['future_return'].mean()
            std_return = group['future_return'].std()
            sharpe_ratio = avg_return / std_return if std_return != 0 else np.nan

            results.append({
                'symbol': symbol,
                'rule': ' AND '.join(rules),
                'max_return': max_return,
                'min_return': min_return,
                'avg_return': avg_return,
                'sharpe_ratio': sharpe_ratio
            })

    # 创建结果DataFrame
    results_df = pd.DataFrame(results)
    # 按夏普比率降序排序
    results_df = results_df.sort_values(by='sharpe_ratio', ascending=False)
    # 保存为CSV
    results_df.to_csv(output_csv, index=False)
    print(f"夏普比率计算结果已保存至 {output_csv}")

def get_prediction_label(prob):
    if prob > 0.7:
        return 1
    elif prob < 0.3:
        return -1
    else:
        return 0

def compute_sharpe_ratio(df, model):
    features = [col for col in df.columns if col not in excluded_columns]
    df['prediction_proba'] = model.predict_proba(df[features])[:, 1]
    df['prediction'] = df['prediction_proba'].apply(get_prediction_label)
    df['strategy_return'] = df['prediction'] * df['return']  # 策略收益
    # 假设无风险利率为 0
    risk_free_rate = 0
    results = []
    # 按 symbol 分组
    for symbol, group in df.groupby('symbol'):
        strategy_std = group['strategy_return'].std()
        market_std = group['return'].std()
        # 处理标准差为零的情况
        if strategy_std == 0:
            strategy_sharpe_ratio = np.nan
        else:
            strategy_sharpe_ratio = (group['strategy_return'].mean() - risk_free_rate) / strategy_std
        if market_std == 0:
            market_sharpe_ratio = np.nan
        else:
            market_sharpe_ratio = (group['return'].mean() - risk_free_rate) / market_std
        results.append({
            'symbol': symbol,
            '策略夏普比率': strategy_sharpe_ratio,
            '市场夏普比率': market_sharpe_ratio
        })
    # 创建表格
    results_df = pd.DataFrame(results)
    print(results_df)
    return results_df

# 实时预测函数
def predict_daily(symbol, model, predict_date, raw_data_dir):
    df = get_futures_data(symbol, raw_data_dir)
    if df is not None and not df.empty:
        df = feature_engineering(df)
        df = df[df['date'] <= pd.Timestamp(predict_date)]  # 筛选到预测日期之前的数据
        if not df.empty:
            features = [col for col in df.columns if col not in excluded_columns]
            prediction = model.predict(df[features])
            print(f"Prediction for {symbol} on {predict_date}: {'Up' if prediction[-1] == 1 else 'Down'}")
        else:
            print(f"No data available for {symbol} on {predict_date}")
    else:
        print(f"No data available for {symbol} on {predict_date}")

# 封装获取多个期货品种数据并处理的函数
def get_and_process_multiple_futures_data(selected_symbols, start_date, end_date, raw_data_dir):
    os.makedirs(feature_data_dir, exist_ok=True)
    combined_path = os.path.join(feature_data_dir, "combined_data.csv")

    if os.path.exists(combined_path):
        print("发现已合并数据，直接加载...")
        combined_data = pd.read_csv(combined_path)
        combined_data['date'] = pd.to_datetime(combined_data['date'])
    else:
        all_data = []
        for symbol in tqdm(selected_symbols):
            print(f"Fetching data for {symbol}...")
            data = get_futures_data(symbol, raw_data_dir)
            if data is not None:
                print("Performing feature engineering...")
                data = feature_engineering(data)
                all_data.append(data)
        combined_data = pd.concat(all_data, ignore_index=True)
        # 根据时间范围筛选数据
        combined_data = combined_data[(combined_data['date'] >= pd.Timestamp(start_date)) & (combined_data['date'] <= pd.Timestamp(end_date))]
        combined_data.to_csv(combined_path, index=False)

    print(f"总数据量: {len(combined_data):,} 行")
    print(f"覆盖品种: {combined_data['symbol'].nunique()} 个")
    print(f"时间范围: {combined_data['date'].min().date()} 至 {combined_data['date'].max().date()}")
    return combined_data

# 主程序
if __name__ == "__main__":
    # Path
    raw_data_dir = "./data/raw_data"
    model_path = './model/xgb_model.joblib'
    feature_data_dir = "./data/feature_data"
    
    # Configuration
    start_date = "2020-01-01"
    end_date = "2024-12-31"
    predict_date = "2025-02-07"
    target_day = 30
    num_symbols = 100  # 选择前100个品种
    excluded_columns = ['date', 'symbol', 'target']
    non_normalize_columns = excluded_columns + ['return', 'rsi', 'skew_7', 'skew_30', 'kurt_7', 'kurt_30', 'turnover_rate']

    # 配置参数
    symbols = get_futures_symbols()
    selected_symbols = symbols[:]
    combined_data = get_and_process_multiple_futures_data(selected_symbols, start_date, end_date, raw_data_dir)

    print("Training model...")
    model = train_model(combined_data)

    print("Backtesting strategy...")
    # compute_sharpe_ratio(combined_data, model)

    split_rules_csv = './model/split_rules.csv'
    sharpe_output_csv = './model/sharpe_by_rules.csv'
    compute_sharpe_by_rules(model, combined_data, split_rules_csv, sharpe_output_csv)

    # for symbol in selected_symbols:
    #     print("Predicting daily movement...")
    #     predict_daily(symbol, model, predict_date, raw_data_dir)
