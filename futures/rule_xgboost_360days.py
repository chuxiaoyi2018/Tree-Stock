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
import heapq
from collections import defaultdict

# 配置全局参数
pd.set_option('display.float_format', lambda x: '%.3f' % x)

# 获取期货代码列表
def get_futures_symbols():
    file_path = os.path.join(raw_data_dir, 'futures_hist_table_em.csv')
    if os.path.exists(file_path):
        futures_list = pd.read_csv(file_path)
    else:
        futures_list = ak.futures_hist_table_em()
        futures_list.to_csv(file_path, encoding='utf-8-sig', index=False)
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
            df.to_csv(file_path, encoding='utf-8-sig', index=False)
            return df
        except Exception as e:
            print(f"Error occurred while fetching data for symbol {symbol}: {e}")
            return None

# 特征工程
def feature_engineering(df):
    # 基本价格特征
    df['return'] = df['close'].pct_change()  # 收益率
    df['momentum'] = df['close'] / df['close'].shift(5) - 1  # 5 日动量
    df['ma_5'] = df['close'].rolling(window=5).mean()  # 5 日均线
    df['ma_10'] = df['close'].rolling(window=10).mean()  # 10 日均线
    df['ma_30'] = df['close'].rolling(window=30).mean()  # 30 日均线
    df['ma_60'] = df['close'].rolling(window=60).mean()  # 60 日均线
    df['ma_180'] = df['close'].rolling(window=180).mean()  # 180 日均线
    df['ma_360'] = df['close'].rolling(window=360).mean()  # 360 日均线
    df['ma_diff_10_30'] = df['ma_10'] - df['ma_30']  # 10 日与 30 日均线差
    df['ma_diff_10_30'] = df['ma_10'] - df['ma_30']  # 10 日与 30 日均线差
    df['ma_diff_10_60'] = df['ma_10'] - df['ma_60']  # 10 日与 60 日均线差
    df['ma_diff_10_180'] = df['ma_10'] - df['ma_180']  # 10 日与 180 日均线差
    df['ma_diff_10_360'] = df['ma_10'] - df['ma_360']  # 10 日与 360 日均线差
    df['ma_diff_30_60'] = df['ma_30'] - df['ma_60']  # 30 日与 60 日均线差
    df['ma_diff_30_180'] = df['ma_30'] - df['ma_180']  # 30 日与 180 日均线差
    df['ma_diff_30_360'] = df['ma_30'] - df['ma_360']  # 30 日与 360 日均线差
    df['ma_diff_60_180'] = df['ma_60'] - df['ma_180']  # 60 日与 180 日均线差
    df['ma_diff_60_360'] = df['ma_60'] - df['ma_360']  # 60 日与 360 日均线差
    df['ma_diff_180_360'] = df['ma_180'] - df['ma_360']  # 180 日与 360 日均线差

    # 不同时间窗口的价格统计特征
    df['max_7'] = df['close'].rolling(window=7).max()  # 7 日最高价
    df['min_7'] = df['close'].rolling(window=7).min()  # 7 日最低价
    df['max_30'] = df['close'].rolling(window=30).max()  # 30 日最高价
    df['min_30'] = df['close'].rolling(window=30).min()  # 30 日最低价
    df['max_60'] = df['close'].rolling(window=60).max()  # 60 日最高价
    df['min_60'] = df['close'].rolling(window=60).min()  # 60 日最低价
    df['max_180'] = df['close'].rolling(window=180).max()  # 180 日最高价
    df['min_180'] = df['close'].rolling(window=180).min()  # 180 日最低价
    df['max_360'] = df['close'].rolling(window=360).max()  # 360 日最高价
    df['min_360'] = df['close'].rolling(window=360).min()  # 360 日最低价

    df['range_7'] = df['max_7'] - df['min_7']  # 7 日价格范围
    df['range_30'] = df['max_30'] - df['min_30']  # 30 日价格范围
    df['range_60'] = df['max_60'] - df['min_60']  # 60 日价格范围
    df['range_180'] = df['max_180'] - df['min_180']  # 180 日价格范围
    df['range_360'] = df['max_360'] - df['min_360']  # 360 日价格范围

    # 量价特征
    df['volume_return'] = df['volume'].pct_change()  # 成交量收益率
    df['price_volume_ratio'] = df['close'] / df['volume']  # 价量比
    df['volume_ma_10'] = df['volume'].rolling(window=10).mean()  # 10 日成交量均线
    df['volume_ma_30'] = df['volume'].rolling(window=30).mean()  # 30 日成交量均线
    df['volume_ma_60'] = df['volume'].rolling(window=60).mean()  # 60 日成交量均线
    df['volume_ma_180'] = df['volume'].rolling(window=180).mean()  # 180 日成交量均线
    df['volume_ma_360'] = df['volume'].rolling(window=360).mean()  # 360 日成交量均线

    df['volume_ma_diff_10_30'] = df['volume_ma_10'] - df['volume_ma_30']  # 10 日与 30 日成交量均线差
    df['volume_ma_diff_10_60'] = df['volume_ma_10'] - df['volume_ma_60']  # 10 日与 60 日成交量均线差
    df['volume_ma_diff_10_180'] = df['volume_ma_10'] - df['volume_ma_180']  # 10 日与 180 日成交量均线差
    df['volume_ma_diff_10_360'] = df['volume_ma_10'] - df['volume_ma_360']  # 10 日与 360 日成交量均线差
    df['volume_ma_diff_30_60'] = df['volume_ma_30'] - df['volume_ma_60']  # 30 日与 60 日成交量均线差
    df['volume_ma_diff_30_180'] = df['volume_ma_30'] - df['volume_ma_180']  # 30 日与 180 日成交量均线差
    df['volume_ma_diff_30_360'] = df['volume_ma_30'] - df['volume_ma_360']  # 30 日与 360 日成交量均线差
    df['volume_ma_diff_60_180'] = df['volume_ma_60'] - df['volume_ma_180']  # 60 日与 180 日成交量均线差
    df['volume_ma_diff_60_360'] = df['volume_ma_60'] - df['volume_ma_360']  # 60 日与 360 日成交量均线差
    df['volume_ma_diff_180_360'] = df['volume_ma_180'] - df['volume_ma_360']  # 180 日与 360 日成交量均线差

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
    df['skew_60'] = df['close'].rolling(window=60).skew()
    df['skew_180'] = df['close'].rolling(window=180).skew()
    df['skew_360'] = df['close'].rolling(window=360).skew()

    # 峰度因子
    df['kurt_7'] = df['close'].rolling(window=7).kurt()
    df['kurt_30'] = df['close'].rolling(window=30).kurt()
    df['kurt_60'] = df['close'].rolling(window=60).kurt()
    df['kurt_180'] = df['close'].rolling(window=180).kurt()
    df['kurt_360'] = df['close'].rolling(window=360).kurt()

    # 波动因子
    df['volatility_10'] = df['close'].rolling(window=10).std()
    df['volatility_30'] = df['close'].rolling(window=30).std()
    df['volatility_60'] = df['close'].rolling(window=60).std()
    df['volatility_180'] = df['close'].rolling(window=180).std()
    df['volatility_360'] = df['close'].rolling(window=360).std()

    # 流动性因子
    df['turnover_rate'] = df['volume'] / (df['open_interest'] if 'open_interest' in df.columns else 1)
    df['liquidity_10'] = df['turnover_rate'].rolling(window=10).mean()
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
    df['money_flow_ma_10'] = df['money_flow'].rolling(window=10).mean()
    df['money_flow_ma_30'] = df['money_flow'].rolling(window=30).mean()

    # 乖离率因子
    df['bias_5'] = (df['close'] - df['ma_5']) / df['ma_5']
    df['bias_10'] = (df['close'] - df['ma_10']) / df['ma_10']
    df['bias_30'] = (df['close'] - df['ma_30']) / df['ma_30']

    # 二分类目标，涨为 1，跌为 0
    df['target'] = (df['close'].shift(-target_day) > df['close']).astype(int)

    # 计算未来 N 天后的收益率
    df['future_diff'] = df['close'].shift(-target_day) - df['close']
    df['future_return'] = df['future_diff'] / df['close']

    # 去除包含缺失值的行
    df = df.replace([np.inf, -np.inf], np.nan)
    # df = df.replace([np.inf, -np.inf], np.nan).dropna()

    # 归一化处理
    columns_to_normalize = [col for col in df.columns if col not in non_normalize_columns]
    # 过滤掉全 NaN 的列
    valid_columns = []
    for col in columns_to_normalize:
        if not df[col].isna().all():
            valid_columns.append(col)
    scaler = MinMaxScaler()
    if len(df) > 0 and len(valid_columns) > 0:
        df.loc[:, valid_columns] = scaler.fit_transform(df[valid_columns])
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
        model = XGBClassifier(
            n_estimators=2000,
            max_depth=5,
            learning_rate=0.05,
            random_state=42,
            eval_metric="error",
            colsample_bytree=0.6,
            colsample_bylevel=0.6,
            gamma=0.1,
            early_stopping_rounds=200
        )
        # 定义验证集
        eval_set = [(X_test, y_test)]
        # 训练模型，传入 eval_set 和 early_stopping_rounds 并开启 verbose
        model.fit(X_train, y_train, eval_set=eval_set, verbose=True)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model Accuracy: {accuracy:.2f}")
        # 保存模型
        os.makedirs('./result', exist_ok=True)
        joblib.dump(model, model_path)
        print(f"Model saved to {model_path}")
        # 调用打印特征重要性的函数
        print_feature_importance(model, features)

    export_split_rules_to_csv(model, features, './result/split_rules.csv')
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
    # 创建保存目录
    if not os.path.exists(feature_importance_dir):
        os.makedirs(feature_importance_dir)
    # 保存排序后的结果到本地
    df_weight_sorted.to_csv(os.path.join(feature_importance_dir, 'feature_importance_weight_top50.csv'))
    df_gain_sorted.to_csv(os.path.join(feature_importance_dir, 'feature_importance_gain_top50.csv'))
    df_cover_sorted.to_csv(os.path.join(feature_importance_dir, 'feature_importance_cover_top50.csv'))

# 导出决策路径到CSV
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
            
            # 处理分裂节点，生成不同分支的条件
            feature = node['feature']
            threshold = node['threshold']
            op = node['op']
            
            # 生成yes分支条件（原条件）
            yes_rule = f"{feature} {op} {threshold}"
            # 生成no分支条件（取反）
            if op == '<':
                no_rule = f"{feature} >= {threshold}"
            elif op == '<=':
                no_rule = f"{feature} > {threshold}"
            else:  # 处理其他可能的操作符（根据实际情况调整）
                no_rule = f"{feature} !{op} {threshold}"
            
            # 递归遍历yes分支
            yes_path = path.copy()
            yes_path.append(yes_rule)
            dfs(node['yes'], yes_path, current_gain + node['gain'], current_cover + node['cover'])
            
            # 递归遍历no分支
            no_path = path.copy()
            no_path.append(no_rule)
            dfs(node['no'], no_path, current_gain + node['gain'], current_cover + node['cover'])
        
        # 从根节点（通常为0）开始遍历
        dfs(0, [], 0.0, 0.0)

    # 转换为 DataFrame
    max_depth = max(len(p['path']) for p in all_paths) if all_paths else 0
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
    
    if data:
        df = pd.DataFrame(data).fillna('')
        df = df.sort_values('total_gain', ascending=False)
        df.to_csv(output_path, encoding='utf-8-sig', index=False)
        print(f"规则已保存至 {output_path}")
    else:
        print("未导出任何规则。")

# 计算夏普比率并保存规则回测结果
def compute_sharpe_by_rules(model, data, split_rules_csv, output_csv='raw_rules.csv'):
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

    all_results = []

    # 遍历每一条规则路径
    for idx, row in tqdm(rules_df.iterrows(), total=rules_df.shape[0], desc="Processing rules"):
        # if row['yes_prob'] > 0.50:
        #     continue
        # 提取规则
        rules = [row[col] for col in rules_df.columns if 'rule_' in col and pd.notna(row[col]) and row[col] != '']
        if not rules:
            continue  # 如果没有规则，跳过

        # 构建布尔遮罩
        mask = pd.Series(True, index=data.index)
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
        current_results = []
        for symbol, group in grouped:
            if group.empty:
                continue

            direction = 1 if row['yes_prob'] > 0.5 else -1
            group['future_return'] = group['future_return'] * direction
            max_return = group['future_return'].max()
            min_return = group['future_return'].min()
            avg_return = group['future_return'].mean()
            std_return = group['future_return'].std()
            sharpe_ratio = avg_return / std_return if std_return != 0 else np.nan
            win_rate = (group['future_return'] > 0).mean()  # 计算胜率
            
            # 计算盈亏比
            profitable_trades = group[group['future_return'] > 0]['future_return']
            losing_trades = group[group['future_return'] < 0]['future_return']
            avg_profit = profitable_trades.mean() if not profitable_trades.empty else 0
            avg_loss = -losing_trades.mean() if not losing_trades.empty else 0
            profit_loss_ratio = avg_profit / avg_loss if avg_loss != 0 else np.nan

            if np.isnan(sharpe_ratio):
                continue  # 忽略夏普比率为 NaN 的情况

            current_results.append({
                'symbol': symbol,
                'rule': ' AND '.join(rules),
                'max_return': max_return,
                'min_return': min_return,
                'avg_return': avg_return,
                'sharpe_ratio': sharpe_ratio,
                'win_rate': win_rate,
                'profit_loss_ratio': profit_loss_ratio,
                'direction': direction
            })
        if current_results:
            current_df = pd.DataFrame(current_results)
            all_results.append(current_df)

    if not all_results:
        print("没有计算出任何夏普比率。")
        return

    results_df = pd.concat(all_results, ignore_index=True)
    results_df.to_csv(output_csv, encoding='utf-8-sig', index=False)
    print(f"规则粗筛与回测结果已保存至 {output_csv}")

    fine_screen_rules(results_df)  # 调用细筛函数

def fine_screen_rules(df):
    """
    细筛规则函数，根据以下条件筛选规则：
    1. 去掉所有 win_rate < 0
    2. 去掉所有 avg_return 小于 0
    3. 去掉所有 min_return 小于 -0.2
    4. 去掉所有 sharpe_ratio 小于 1
    5. 去掉所有 profit_loss_ratio 小于 2
    然后按照 sharpe_ratio 来排列规则
    :param df: 包含回测结果的DataFrame
    """
    df = df.groupby(['rule', 'direction']).agg({
        'avg_return': 'mean',
        'sharpe_ratio': 'mean',
        'win_rate': 'mean',
        'profit_loss_ratio': 'min',
        'min_return': 'min'
    }).reset_index()

    # 筛选条件
    df = df[df['win_rate'] >= 0.5]
    df = df[df['avg_return'] >= 0]
    df = df[df['profit_loss_ratio'] >= 2]
    df = df[df['min_return'] >= -0.2]
    df = df[df['sharpe_ratio'] >= 1.5]
    # 按照 sharpe_ratio 降序排列
    df = df.sort_values(by='sharpe_ratio', ascending=False)

    df.to_csv(fine_rules_csv, encoding='utf-8-sig', index=False)
    print(f"规则细筛结果已保存至 {fine_rules_csv}")

# 生成交易信号
def generate_signals(data, fine_rules_csv, output_csv='signals.csv'):
    """
    根据细筛后的规则在验证集上生成交易信号
    :param data: 包含特征和target的DataFrame
    :param fine_rules_csv: 细筛后的规则CSV文件路径
    :param output_csv: 输出信号CSV文件路径
    """
    # 读取细筛后的规则
    rules_df = pd.read_csv(fine_rules_csv)
    
    # 初始化信号列表
    signals = []
    
    # 遍历每条规则
    for idx, rule in rules_df.iterrows():
        # 提取规则
        rules = rule['rule'].split(' AND ')
        
        # 构建布尔遮罩
        mask = pd.Series(True, index=data.index)
        for condition in rules:
            # 解析条件，假设条件格式为 "feature operator threshold"
            match = re.match(r'(\w+)\s*([<>]=?)\s*([\d\.eE+-]+)', condition.strip())
            if match:
                feature, operator, threshold = match.groups()
                threshold = float(threshold)
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
        
        # 获取满足条件的信号
        signal_data = data[mask].copy()
        if len(np.unique([s[:2] for s in signal_data['symbol'].tolist()])) < min_unique_signal_num:continue

        signal_data['rule'] = rule['rule']
        signal_data['symbol'] = signal_data['symbol']
        signal_data['date'] = signal_data['date']
        
        # 决定交易方向，依据yes_prob
        # 假设yes_prob > 0.5时开多（买入），否则开空（卖出)
        signal_data['direction'] = rule['direction']  # 示例逻辑，具体依据可调整
        
        # 选择所需字段
        signal = signal_data[['rule', 'symbol', 'direction', 'date', 'future_return']]
        signals.append(signal)
    
    if signals:
        signals_df = pd.concat(signals, ignore_index=True)
        # 排序
        signals_df = signals_df.sort_values(by=['date', 'symbol']).reset_index(drop=True)
        # 保存信号到CSV
        signals_df.to_csv(output_csv, encoding='utf-8-sig', index=False)
        print(f"交易信号已保存至 {output_csv}")
    else:
        print("未生成任何交易信号。")
    
    return signals_df

# 计算回测性能指标
def compute_backtest_metrics(data, signals_df, output_metrics_csv='backtest_metrics.csv'):
    """
    计算回测性能指标
    :param data: 包含特征和target的DataFrame
    :param signals_df: 生成的交易信号DataFrame
    :param output_metrics_csv: 输出的性能指标CSV文件路径
    """
    # 合并信号与数据
    data_with_signals = pd.merge(data, signals_df, on=['symbol', 'date', 'future_return'], how='inner')
    
    # 初始化结果列表
    metrics = []
    
    # 遍历每个规则
    for rule, group in data_with_signals.groupby(['rule']):
        symbol_metrics = {}
        symbol_metrics['rule'] = rule[0]
        direction = group['direction'].min()
        symbol_metrics['direction'] = direction
        group['future_return'] = group['future_return'] * direction
        symbol_metrics['avg_return'] = group['future_return'].mean()
        symbol_metrics['sharpe_ratio'] = group['future_return'].mean() / group['future_return'].std() if group['future_return'].std() != 0 else np.nan
        symbol_metrics['win_rate'] = (group['future_return'] > 0).mean()
        symbol_metrics['min_return'] = group['future_return'].min()

        # 计算盈亏比
        profitable_trades = group[group['future_return'] > 0]['future_return']
        losing_trades = group[group['future_return'] < 0]['future_return']
        avg_profit = profitable_trades.mean() if not profitable_trades.empty else 0
        avg_loss = -losing_trades.mean() if not losing_trades.empty else 0
        symbol_metrics['profit_loss_ratio'] = avg_profit / avg_loss if avg_loss != 0 else np.nan
        metrics.append(symbol_metrics)

    if metrics:
        metrics_df = pd.DataFrame(metrics)
        # 按照sharpe_ratio降序排列
        metrics_df = metrics_df[metrics_df['sharpe_ratio'] >= 1.5]
        metrics_df = metrics_df.sort_values(by='sharpe_ratio', ascending=False).reset_index(drop=True)
        # 保存指标到CSV
        metrics_df.to_csv(output_metrics_csv, encoding='utf-8-sig', index=False)
        print(f"回测性能指标已保存至 {output_metrics_csv}")
    else:
        print("未计算到任何回测性能指标。")

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
        combined_data.to_csv(combined_path, encoding='utf-8-sig', index=False)

    print(f"总数据量: {len(combined_data):,} 行")
    print(f"覆盖品种: {combined_data['symbol'].nunique()} 个")
    print(f"时间范围: {combined_data['date'].min().date()} 至 {combined_data['date'].max().date()}")
    return combined_data

# 主程序
if __name__ == "__main__":
    # Path
    raw_data_dir = "./data/raw_data"
    feature_data_dir = "./data/feature_data"
    model_path = './result/xgb_model.joblib'
    feature_importance_dir = './result/feature_importance/'
    split_rules_csv = './result/split_rules.csv'
    raw_rules_csv = './result/raw_rules.csv'
    fine_rules_csv = './result/fine_rules.csv'
    signals_csv = './result/signals.csv'
    backtest_metrics_csv = './result/backtest_metrics.csv'
    
    # Configuration
    start_date = "2020-01-01"
    end_date = "2023-12-31"
    backtest_start_date = "2024-01-01"  # 回测开始日期
    backtest_end_date = "2024-12-31"    # 回测结束日期
    target_day = 30
    num_symbols = 100  # 选择前100个品种
    min_unique_signal_num = 10
    excluded_columns = ['date', 'symbol', 'target', 'future_return', 'future_diff']
    non_normalize_columns = excluded_columns + ['return', 'rsi', 'skew_7', 'skew_30', 'kurt_7', 'kurt_30', 'turnover_rate']

    # 配置参数
    symbols = get_futures_symbols()
    selected_symbols = symbols[:]
    combined_data = get_and_process_multiple_futures_data(selected_symbols, start_date, end_date, raw_data_dir)
    
    train_data = combined_data[(combined_data['date'] >= pd.Timestamp(start_date)) & (combined_data['date'] <= pd.Timestamp(end_date))]
    test_data = combined_data[(combined_data['date'] >= pd.Timestamp(backtest_start_date)) & (combined_data['date'] <= pd.Timestamp(backtest_end_date))]

    print("Training model...")
    model = train_model(train_data)

    print("Backtesting strategy...")
    # compute_sharpe_by_rules(model, train_data, split_rules_csv, raw_rules_csv)

    print("生成交易信号...")
    signals_df = generate_signals(
        data=test_data,
        fine_rules_csv=fine_rules_csv,
        output_csv=signals_csv
    )

    print("计算回测性能指标...")
    compute_backtest_metrics(
        data=test_data,
        signals_df=signals_df,
        output_metrics_csv=backtest_metrics_csv
    )
