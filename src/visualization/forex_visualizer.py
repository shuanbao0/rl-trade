"""
Forex专用可视化器

专门为外汇交易数据和分析提供可视化功能。
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import logging

from .base_visualizer import BaseVisualizer


class ForexVisualizer(BaseVisualizer):
    """
    Forex专用可视化器
    
    专门处理外汇交易相关的数据可视化，包括：
    - 点数(Pips)分析
    - 货币对价格走势
    - 趋势分析
    - 支撑阻力位
    - 交易时段分析
    - 点差和滑点分析
    """
    
    def __init__(self, **kwargs):
        """初始化Forex可视化器"""
        super().__init__(**kwargs)
        self.logger = logging.getLogger('ForexVisualizer')
        
        # Forex专用颜色
        self.forex_colors = {
            'bullish': '#26A69A',      # 看涨绿色
            'bearish': '#EF5350',      # 看跌红色
            'doji': '#FFA726',         # 十字星橙色
            'support': '#2196F3',      # 支撑蓝色
            'resistance': '#F44336',   # 阻力红色
            'trend_up': '#4CAF50',     # 上升趋势
            'trend_down': '#FF5722',   # 下降趋势
            'sideways': '#9E9E9E'      # 横盘整理
        }
    
    def plot_candlestick_chart(self, 
                              ohlc_data: pd.DataFrame,
                              actions: Optional[List[float]] = None,
                              title: str = "外汇蜡烛图",
                              show_volume: bool = False) -> List[str]:
        """
        绘制蜡烛图
        
        Args:
            ohlc_data: OHLC数据，包含Open, High, Low, Close, Volume(可选)
            actions: 交易动作序列
            title: 图表标题
            show_volume: 是否显示成交量
            
        Returns:
            List[str]: 保存的文件路径
        """
        # 创建子图
        if show_volume and 'Volume' in ohlc_data.columns:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), 
                                         height_ratios=[3, 1], sharex=True)
        else:
            fig, ax1 = plt.subplots(figsize=(16, 8))
        
        # 绘制蜡烛图
        self._draw_candlesticks(ax1, ohlc_data)
        
        # 添加交易信号
        if actions is not None:
            self._add_trading_signals(ax1, ohlc_data, actions)
        
        # 添加技术指标
        self._add_technical_indicators(ax1, ohlc_data)
        
        ax1.set_title(title, fontsize=16, fontweight='bold')
        ax1.set_ylabel('Price')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # 绘制成交量（如果需要）
        if show_volume and 'Volume' in ohlc_data.columns:
            self._draw_volume(ax2, ohlc_data)
            ax2.set_ylabel('Volume')
            ax2.set_xlabel('Time')
        else:
            ax1.set_xlabel('Time')
        
        return self.save_figure(fig, 'forex_candlestick', 'forex')
    
    def plot_pip_analysis(self, 
                         prices: List[float],
                         actions: List[float],
                         pip_size: float = 0.0001,
                         currency_pair: str = "EURUSD") -> List[str]:
        """
        绘制点数分析
        
        Args:
            prices: 价格序列
            actions: 交易动作
            pip_size: 点大小
            currency_pair: 货币对名称
            
        Returns:
            List[str]: 保存的文件路径
        """
        fig, axes = self.create_figure(figsize=(16, 12), subplots=(2, 2))
        
        # 计算点数变化
        pip_changes = [((prices[i] - prices[i-1]) / pip_size) 
                      for i in range(1, len(prices))]
        
        # 计算点数收益（考虑交易方向）
        pip_profits = []
        for i in range(1, len(prices)):
            if i-1 < len(actions):
                action = actions[i-1]
                pip_change = pip_changes[i-1]
                pip_profit = pip_change * action
                pip_profits.append(pip_profit)
        
        time_steps = np.arange(len(pip_changes))
        
        # 1. 点数变化时间序列
        axes[0].plot(time_steps, pip_changes, 
                    color=self.colors['primary'], linewidth=1, alpha=0.7)
        axes[0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[0].fill_between(time_steps, pip_changes, 0,
                           where=np.array(pip_changes) >= 0,
                           color=self.forex_colors['bullish'], alpha=0.3)
        axes[0].fill_between(time_steps, pip_changes, 0,
                           where=np.array(pip_changes) < 0,
                           color=self.forex_colors['bearish'], alpha=0.3)
        axes[0].set_title(f'{currency_pair} Pip Changes', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Time Step')
        axes[0].set_ylabel('Pip Changes')
        axes[0].grid(True, alpha=0.3)
        
        # 2. 点数分布直方图
        axes[1].hist(pip_changes, bins=50, alpha=0.7, 
                    color=self.colors['info'], edgecolor='black')
        axes[1].axvline(np.mean(pip_changes), color='red', 
                       linestyle='--', linewidth=2, 
                       label=f'均值: {np.mean(pip_changes):.2f} pips')
        axes[1].axvline(np.median(pip_changes), color='orange', 
                       linestyle='--', linewidth=2, 
                       label=f'中位数: {np.median(pip_changes):.2f} pips')
        axes[1].set_title('点数变化分布', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('点数')
        axes[1].set_ylabel('频次')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # 3. 累积点数收益
        if pip_profits:
            cumulative_pips = np.cumsum(pip_profits)
            axes[2].plot(time_steps[:len(cumulative_pips)], cumulative_pips, 
                        color=self.colors['success'], linewidth=2)
            axes[2].axhline(y=0, color='black', linestyle='--', alpha=0.5)
            axes[2].set_title('累积点数收益', fontsize=14, fontweight='bold')
            axes[2].set_xlabel('时间步')
            axes[2].set_ylabel('累积点数')
            axes[2].grid(True, alpha=0.3)
            
            # 添加统计信息
            final_pips = cumulative_pips[-1]
            max_pips = max(cumulative_pips)
            min_pips = min(cumulative_pips)
            
            stats_text = (
                f'点数统计:\n'
                f'最终点数: {final_pips:.1f}\n'
                f'最大盈利: {max_pips:.1f}\n'
                f'最大亏损: {min_pips:.1f}\n'
                f'平均每笔: {np.mean(pip_profits):.2f}'
            )
            
            axes[2].text(0.02, 0.98, stats_text,
                        transform=axes[2].transAxes,
                        verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        # 4. 点数收益分布
        if pip_profits:
            axes[3].hist(pip_profits, bins=30, alpha=0.7,
                        color=self.colors['secondary'], edgecolor='black')
            axes[3].axvline(0, color='black', linestyle='--', alpha=0.5)
            axes[3].set_title('单笔点数收益分布', fontsize=14, fontweight='bold')
            axes[3].set_xlabel('点数收益')
            axes[3].set_ylabel('频次')
            axes[3].grid(True, alpha=0.3)
            
            # 计算胜率
            winning_trades = len([p for p in pip_profits if p > 0])
            total_trades = len(pip_profits)
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            axes[3].text(0.02, 0.98, f'点数胜率: {win_rate:.2%}',
                        transform=axes[3].transAxes,
                        verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        return self.save_figure(fig, f'{currency_pair.lower()}_pip_analysis', 'forex')
    
    def plot_trend_analysis(self, 
                           prices: List[float],
                           trend_periods: List[int] = [10, 20, 50],
                           currency_pair: str = "EURUSD") -> List[str]:
        """
        绘制趋势分析
        
        Args:
            prices: 价格序列
            trend_periods: 趋势分析周期
            currency_pair: 货币对名称
            
        Returns:
            List[str]: 保存的文件路径
        """
        fig, axes = self.create_figure(figsize=(16, 10), subplots=(2, 1))
        
        time_steps = np.arange(len(prices))
        df = pd.DataFrame({'price': prices})
        
        # 1. 价格走势和移动平均线
        axes[0].plot(time_steps, prices, color='black', linewidth=1, 
                    label='价格', alpha=0.8)
        
        colors = [self.colors['primary'], self.colors['success'], self.colors['warning']]
        
        for i, period in enumerate(trend_periods):
            if len(prices) > period:
                ma = df['price'].rolling(window=period).mean()
                axes[0].plot(time_steps, ma, color=colors[i % len(colors)], 
                           linewidth=2, label=f'MA{period}', alpha=0.8)
        
        axes[0].set_title(f'{currency_pair} 趋势分析', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('时间步')
        axes[0].set_ylabel('价格')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 2. 趋势强度分析
        trend_strength = []
        trend_direction = []
        
        window = 20
        for i in range(window, len(prices)):
            price_window = prices[i-window:i]
            
            # 计算线性回归斜率作为趋势强度
            x = np.arange(len(price_window))
            slope, intercept = np.polyfit(x, price_window, 1)
            
            # 标准化斜率
            price_range = max(price_window) - min(price_window)
            normalized_slope = slope / price_range if price_range > 0 else 0
            
            trend_strength.append(abs(normalized_slope))
            trend_direction.append(1 if slope > 0 else -1)
        
        trend_steps = time_steps[window:]
        
        # 绘制趋势强度
        axes[1].fill_between(trend_steps, 0, trend_strength,
                           color=self.colors['info'], alpha=0.7, label='趋势强度')
        
        # 用颜色标记趋势方向
        for i, (step, strength, direction) in enumerate(zip(trend_steps, trend_strength, trend_direction)):
            color = self.forex_colors['trend_up'] if direction > 0 else self.forex_colors['trend_down']
            axes[1].bar(step, strength, width=1, color=color, alpha=0.6)
        
        axes[1].set_title('趋势强度和方向', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('时间步')
        axes[1].set_ylabel('趋势强度')
        axes[1].grid(True, alpha=0.3)
        
        # 添加趋势统计
        if trend_strength:
            up_trend_pct = len([d for d in trend_direction if d > 0]) / len(trend_direction)
            avg_strength = np.mean(trend_strength)
            
            stats_text = (
                f'趋势统计:\n'
                f'上升趋势: {up_trend_pct:.1%}\n'
                f'下降趋势: {1-up_trend_pct:.1%}\n'
                f'平均强度: {avg_strength:.4f}'
            )
            
            axes[1].text(0.02, 0.98, stats_text,
                        transform=axes[1].transAxes,
                        verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        
        return self.save_figure(fig, f'{currency_pair.lower()}_trend_analysis', 'forex')
    
    def plot_support_resistance(self, 
                               ohlc_data: pd.DataFrame,
                               lookback_period: int = 20,
                               min_strength: int = 3) -> List[str]:
        """
        绘制支撑阻力位分析
        
        Args:
            ohlc_data: OHLC数据
            lookback_period: 回望周期
            min_strength: 最小强度（触及次数）
            
        Returns:
            List[str]: 保存的文件路径
        """
        fig, ax = self.create_figure(figsize=(16, 10), subplots=(1, 1))
        ax = ax[0]
        
        # 绘制价格线
        time_index = np.arange(len(ohlc_data))
        ax.plot(time_index, ohlc_data['Close'], color='black', 
               linewidth=1, label='收盘价', alpha=0.8)
        
        # 识别支撑和阻力位
        support_levels, resistance_levels = self._find_support_resistance(
            ohlc_data, lookback_period, min_strength)
        
        # 绘制支撑位
        for level, strength, start_idx, end_idx in support_levels:
            ax.hlines(level, start_idx, end_idx, 
                     colors=self.forex_colors['support'], 
                     linestyles='solid', linewidth=2, alpha=0.7)
            ax.text(end_idx, level, f'支撑 ({strength})', 
                   fontsize=10, color=self.forex_colors['support'],
                   verticalalignment='bottom')
        
        # 绘制阻力位
        for level, strength, start_idx, end_idx in resistance_levels:
            ax.hlines(level, start_idx, end_idx, 
                     colors=self.forex_colors['resistance'], 
                     linestyles='solid', linewidth=2, alpha=0.7)
            ax.text(end_idx, level, f'阻力 ({strength})', 
                   fontsize=10, color=self.forex_colors['resistance'],
                   verticalalignment='top')
        
        # 添加突破标记
        self._mark_breakouts(ax, ohlc_data, support_levels, resistance_levels)
        
        ax.set_title('支撑阻力位分析', fontsize=14, fontweight='bold')
        ax.set_xlabel('时间步')
        ax.set_ylabel('价格')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return self.save_figure(fig, 'support_resistance_analysis', 'forex')
    
    def plot_trading_session_analysis(self, 
                                     ohlc_data: pd.DataFrame,
                                     timezone: str = "UTC") -> List[str]:
        """
        绘制交易时段分析
        
        Args:
            ohlc_data: 包含时间索引的OHLC数据
            timezone: 时区
            
        Returns:
            List[str]: 保存的文件路径
        """
        if not isinstance(ohlc_data.index, pd.DatetimeIndex):
            self.logger.warning("需要时间索引进行交易时段分析")
            return []
        
        fig, axes = self.create_figure(figsize=(16, 12), subplots=(2, 2))
        
        # 提取时间信息
        ohlc_data['hour'] = ohlc_data.index.hour
        ohlc_data['day_of_week'] = ohlc_data.index.dayofweek
        
        # 1. 按小时分析波动率
        hourly_volatility = ohlc_data.groupby('hour').apply(
            lambda x: ((x['High'] - x['Low']) / x['Close']).mean())
        
        axes[0].bar(hourly_volatility.index, hourly_volatility.values,
                   color=self.colors['primary'], alpha=0.7)
        axes[0].set_title('小时波动率分布', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('小时 (UTC)')
        axes[0].set_ylabel('平均波动率')
        axes[0].grid(True, alpha=0.3)
        
        # 标记主要交易时段
        sessions = {
            'Asian': (0, 9),
            'London': (8, 16),
            'New York': (13, 22)
        }
        
        for session_name, (start, end) in sessions.items():
            axes[0].axvspan(start, end, alpha=0.2, label=session_name)
        axes[0].legend()
        
        # 2. 按星期分析
        day_names = ['周一', '周二', '周三', '周四', '周五', '周六', '周日']
        daily_returns = ohlc_data.groupby('day_of_week')['Close'].apply(
            lambda x: x.pct_change().mean() * 100)
        
        colors = [self.colors['profit'] if r >= 0 else self.colors['loss'] 
                 for r in daily_returns.values]
        axes[1].bar(range(7), daily_returns.values, color=colors, alpha=0.7)
        axes[1].set_xticks(range(7))
        axes[1].set_xticklabels(day_names)
        axes[1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[1].set_title('星期平均收益率', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('星期')
        axes[1].set_ylabel('平均收益率 (%)')
        axes[1].grid(True, alpha=0.3)
        
        # 3. 成交量时段分析（如果有成交量数据）
        if 'Volume' in ohlc_data.columns:
            hourly_volume = ohlc_data.groupby('hour')['Volume'].mean()
            axes[2].bar(hourly_volume.index, hourly_volume.values,
                       color=self.colors['info'], alpha=0.7)
            axes[2].set_title('小时平均成交量', fontsize=14, fontweight='bold')
            axes[2].set_xlabel('小时 (UTC)')
            axes[2].set_ylabel('平均成交量')
        else:
            axes[2].axis('off')
            axes[2].text(0.5, 0.5, '无成交量数据', 
                        transform=axes[2].transAxes,
                        ha='center', va='center', fontsize=16)
        
        # 4. 时段热力图
        pivot_data = ohlc_data.pivot_table(
            values='Close', 
            index='hour', 
            columns='day_of_week', 
            aggfunc=lambda x: x.pct_change().mean()
        )
        
        sns.heatmap(pivot_data, annot=True, fmt='.4f', 
                   cmap='RdYlGn', center=0, ax=axes[3],
                   xticklabels=['周一', '周二', '周三', '周四', '周五', '周六', '周日'],
                   yticklabels=range(24))
        axes[3].set_title('时段收益率热力图', fontsize=14, fontweight='bold')
        axes[3].set_xlabel('星期')
        axes[3].set_ylabel('小时')
        
        return self.save_figure(fig, 'trading_session_analysis', 'forex')
    
    def _draw_candlesticks(self, ax, ohlc_data):
        """绘制蜡烛图"""
        for i, (idx, row) in enumerate(ohlc_data.iterrows()):
            open_price, high_price, low_price, close_price = row['Open'], row['High'], row['Low'], row['Close']
            
            # 确定颜色
            if close_price >= open_price:
                color = self.forex_colors['bullish']
                lower, upper = open_price, close_price
            else:
                color = self.forex_colors['bearish']
                lower, upper = close_price, open_price
            
            # 绘制影线
            ax.plot([i, i], [low_price, high_price], 
                   color='black', linewidth=1, alpha=0.8)
            
            # 绘制实体
            ax.fill_between([i-0.3, i+0.3], [lower, lower], [upper, upper],
                           color=color, alpha=0.8)
    
    def _add_trading_signals(self, ax, ohlc_data, actions):
        """添加交易信号标记"""
        buy_signals = []
        sell_signals = []
        
        for i, action in enumerate(actions):
            if i < len(ohlc_data):
                if action > 0.5:
                    buy_signals.append((i, ohlc_data.iloc[i]['Low']))
                elif action < -0.5:
                    sell_signals.append((i, ohlc_data.iloc[i]['High']))
        
        if buy_signals:
            x_buy, y_buy = zip(*buy_signals)
            ax.scatter(x_buy, y_buy, color='green', marker='^', 
                      s=100, label='买入信号', zorder=5)
        
        if sell_signals:
            x_sell, y_sell = zip(*sell_signals)
            ax.scatter(x_sell, y_sell, color='red', marker='v', 
                      s=100, label='卖出信号', zorder=5)
    
    def _add_technical_indicators(self, ax, ohlc_data):
        """添加技术指标"""
        if len(ohlc_data) > 20:
            # 简单移动平均
            sma20 = ohlc_data['Close'].rolling(window=20).mean()
            ax.plot(range(len(sma20)), sma20, color='blue', 
                   linewidth=1, label='SMA20', alpha=0.7)
        
        if len(ohlc_data) > 50:
            sma50 = ohlc_data['Close'].rolling(window=50).mean()
            ax.plot(range(len(sma50)), sma50, color='orange', 
                   linewidth=1, label='SMA50', alpha=0.7)
    
    def _draw_volume(self, ax, ohlc_data):
        """绘制成交量"""
        volumes = ohlc_data['Volume'].values
        colors = [self.forex_colors['bullish'] if ohlc_data.iloc[i]['Close'] >= ohlc_data.iloc[i]['Open'] 
                 else self.forex_colors['bearish'] for i in range(len(ohlc_data))]
        
        ax.bar(range(len(volumes)), volumes, color=colors, alpha=0.7)
        ax.grid(True, alpha=0.3)
    
    def _find_support_resistance(self, ohlc_data, lookback_period, min_strength):
        """识别支撑和阻力位"""
        support_levels = []
        resistance_levels = []
        
        highs = ohlc_data['High'].values
        lows = ohlc_data['Low'].values
        
        # 简化的支撑阻力识别算法
        for i in range(lookback_period, len(ohlc_data) - lookback_period):
            # 检查是否是局部高点
            if highs[i] == max(highs[i-lookback_period:i+lookback_period]):
                level = highs[i]
                # 计算这个水平的强度
                touches = sum([1 for h in highs if abs(h - level) < level * 0.001])
                if touches >= min_strength:
                    resistance_levels.append((level, touches, max(0, i-50), min(len(ohlc_data), i+50)))
            
            # 检查是否是局部低点
            if lows[i] == min(lows[i-lookback_period:i+lookback_period]):
                level = lows[i]
                touches = sum([1 for l in lows if abs(l - level) < level * 0.001])
                if touches >= min_strength:
                    support_levels.append((level, touches, max(0, i-50), min(len(ohlc_data), i+50)))
        
        return support_levels, resistance_levels
    
    def _mark_breakouts(self, ax, ohlc_data, support_levels, resistance_levels):
        """标记突破点"""
        # 简化的突破标记
        for level, _, start_idx, end_idx in resistance_levels:
            for i in range(start_idx, min(end_idx, len(ohlc_data))):
                if ohlc_data.iloc[i]['Close'] > level:
                    ax.scatter(i, ohlc_data.iloc[i]['Close'], 
                             color='green', marker='*', s=100, alpha=0.8)
                    break
        
        for level, _, start_idx, end_idx in support_levels:
            for i in range(start_idx, min(end_idx, len(ohlc_data))):
                if ohlc_data.iloc[i]['Close'] < level:
                    ax.scatter(i, ohlc_data.iloc[i]['Close'], 
                             color='red', marker='*', s=100, alpha=0.8)
                    break