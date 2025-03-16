import backtrader as bt
import pandas as pd
import numpy as np
from joblib import load
from datetime import datetime, timedelta
from collections import defaultdict

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

class EnsembleStrategy(bt.Strategy):
    params = (
        ('model_count', 10),
        ('holding_period', 14),
        ('position_ratio', 1.0),
    )
    
    def __init__(self):
        self.trade_count = 0
        self.setsizer(bt.sizers.PercentSizer(percents=self.p.position_ratio))

        self.models = []
        for i in range(self.p.model_count):
            model_data = load(f'models/lr_model_{i}.joblib')
            self.models.append({
                'model': model_data['model'],
                'features': model_data['features']
            })
        
        # 持仓状态: {'entry_bar': int}
        self.position_info = dict()

    def notify_trade(self, trade):
        """交易关闭时触发"""
        if trade.isclosed:
            self.trade_count += 1

    def next(self):
        data = self.datas[0]  # 现在只有一个品种
        
        # 平仓逻辑
        if 'entry_bar' in self.position_info and (len(self) - self.position_info['entry_bar']) >= self.p.holding_period:
            self.close(data=data)  # 平仓当前品种
            self.position_info.clear()
            return
                
        # 跳过已有持仓的品种
        if self.getposition(data).size != 0:
            return
                
        # 获取当前品种的数据
        current_data = {}
        for f in set().union(*[m['features'] for m in self.models]):
            current_data[f] = getattr(data, f)[0]
        
        # 模型预测
        predictions = []
        for model_info in self.models:
            try:
                current_df = pd.DataFrame([current_data])[model_info['features']]
                pred = model_info['model'].predict(current_df)[0]
                predictions.append(pred)
            except Exception as e:
                print(f"模型预测错误: {str(e)}")
                continue
        
        # 判断信号
        long_signal = sum(p == 3 for p in predictions) >= 2
        short_signal = sum(p == -3 for p in predictions) >= 2
        
        # 执行交易并记录持仓
        if long_signal:
            self.buy(data=data)
            self.position_info['entry_bar'] = len(self)
        elif short_signal:
            self.sell(data=data)
            self.position_info['entry_bar'] = len(self)

def load_data():
    df = pd.read_csv('../data/feature_data/combined_data.csv', parse_dates=['date'])
    df = df[df['date'].dt.year == 2023]
    mandatory_cols = ['date', 'open', 'high', 'low', 'close', 'volume', 'symbol']
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    df.rename(columns={'open_interest': 'openinterest'}, inplace=True)
    feature_cols = [col for col in df.columns if col not in mandatory_cols + ['openinterest']]
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
    df = df.sort_values(['symbol', 'date']).set_index('date')

    datafeeds = []
    for symbol, group in df.groupby('symbol'):
        group = group.drop(columns=['symbol'])
        
        # 获取所有的特征列
        extra_cols = [col for col in group.columns if col not in ['open','high','low','close','volume','openinterest']]
        
        # 定义 lines，包含基础列和特征列
        lines = ('open', 'high', 'low', 'close', 'volume', 'openinterest') + tuple(extra_cols)
        
        # 定义 params，映射数据列
        params_list = [
            ('datetime', None),
            ('open', 'open'),
            ('high', 'high'),
            ('low', 'low'),
            ('close', 'close'),
            ('volume', 'volume'),
            ('openinterest', 'openinterest'),
        ] + [(col, col) for col in extra_cols]
        
        # 使用 type 动态创建包含自定义 lines 和 params 的类
        DynamicData = type('DynamicData', (bt.feeds.PandasData,), {
            'lines': lines,
            'params': tuple(params_list),
        })
        
        # 创建数据对象并设置名称
        data = DynamicData(dataname=group)
        data._name = symbol  # 通过 _name 存储品种名称
        datafeeds.append(data)
    
    return datafeeds
if __name__ == '__main__':
    datafeeds = load_data()
    results = []
    
    for data in datafeeds:
        cerebro = bt.Cerebro()
        cerebro.adddata(data, name=data._name)
        cerebro.addstrategy(EnsembleStrategy)
        cerebro.broker.setcash(100000)
        cerebro.broker.setcommission(commission=0.0001)
        
        # 运行单个品种回测
        print(f"\n开始回测品种: {data._name}")
        strategies = cerebro.run()  # 获取策略实例列表
        strat = strategies[0]       # 提取策略实例
        
        # 记录结果
        final_value = cerebro.broker.getvalue()
        results.append({
            'symbol': data._name,
            'final_value': final_value,
            'return_pct': (final_value / 100000 - 1) * 100,
            'trade_count': strat.trade_count  # 新增交易次数
        })
        
        # 输出当前品种结果
        print(f"交易次数: {strat.trade_count} 手")  # 新增输出
        print(f"最终资金: {final_value:.2f}")
        print(f"收益率: {results[-1]['return_pct']:.2f}%")
        pos = cerebro.getbroker().getposition(data)
        if pos.size != 0:
            print(f'未平仓头寸: {pos.size}')
    
    # 汇总所有品种结果
    print("\n全品种回测汇总:")
    results_df = pd.DataFrame(results)
    results_df.to_csv("symbol_backtrade.csv", index=False)
    print(results_df)
    print("\n统计摘要:")
    print(f"平均交易次数: {results_df['trade_count'].mean():.1f} 手")
    print(f"最终资金均值: {results_df['final_value'].mean():.2f}")
    print(f"平均收益率: {results_df['return_pct'].mean():.2f}%")