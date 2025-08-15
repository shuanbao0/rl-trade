"""
评估结果可视化器

专门用于可视化模型评估结果和性能分析。
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import logging
from scipy import stats

from .base_visualizer import BaseVisualizer


class EvaluationVisualizer(BaseVisualizer):
    """
    评估结果可视化器
    
    专门处理模型评估相关的数据可视化，包括：
    - Episode性能分析
    - 交易信号质量
    - 风险收益分析
    - 回测结果展示
    - 策略有效性验证
    """
    
    def __init__(self, **kwargs):
        """初始化评估可视化器"""
        super().__init__(**kwargs)
        # 使用统一日志系统
        from ..utils.logger import get_logger
        self.logger = get_logger('EvaluationVisualizer')
    
    def plot_episode_performance(self, 
                                episode_data: List[Dict[str, Any]],
                                show_individual: bool = True) -> List[str]:
        """
        绘制各个episode的性能表现
        
        Args:
            episode_data: 每个episode的数据字典列表
                格式: [{'episode': 1, 'reward': 0.5, 'return': 2.1, 'steps': 100, ...}, ...]
            show_individual: 是否显示单个episode详情
            
        Returns:
            List[str]: 保存的文件路径
        """
        if not episode_data:
            self.logger.warning("没有提供episode数据")
            return []
        
        fig, axes = self.create_figure(figsize=(16, 12), subplots=(3, 2))
        
        # 提取数据
        episodes = [d.get('episode', i+1) for i, d in enumerate(episode_data)]
        rewards = [d.get('reward', 0) for d in episode_data]
        returns = [d.get('return', 0) for d in episode_data]
        steps = [d.get('steps', 0) for d in episode_data]
        final_values = [d.get('final_value', 10000) for d in episode_data]
        
        # 1. Episode奖励表现
        axes[0].bar(episodes, rewards, 
                   color=[self.colors['profit'] if r >= 0 else self.colors['loss'] for r in rewards],
                   alpha=0.7, edgecolor='black', linewidth=0.5)
        axes[0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[0].set_title('Episode Reward Performance', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Episode')
        axes[0].set_ylabel('Cumulative Reward')
        axes[0].grid(True, alpha=0.3)
        
        # 2. Episode收益率表现
        axes[1].bar(episodes, returns, 
                   color=[self.colors['profit'] if r >= 0 else self.colors['loss'] for r in returns],
                   alpha=0.7, edgecolor='black', linewidth=0.5)
        axes[1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[1].set_title('Episode Return Performance', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Episode')
        axes[1].set_ylabel('Return (%)')
        axes[1].grid(True, alpha=0.3)
        
        # 3. Episode步数分布
        axes[2].bar(episodes, steps, color=self.colors['info'], 
                   alpha=0.7, edgecolor='black', linewidth=0.5)
        axes[2].set_title('Episode步数', fontsize=14, fontweight='bold')
        axes[2].set_xlabel('Episode')
        axes[2].set_ylabel('步数')
        axes[2].grid(True, alpha=0.3)
        
        # 4. 收益-奖励相关性
        if len(rewards) > 1 and len(returns) > 1:
            axes[3].scatter(rewards, returns, 
                           color=self.colors['secondary'], alpha=0.7, s=60)
            
            # 添加趋势线
            if len(rewards) > 2:
                z = np.polyfit(rewards, returns, 1)
                trend_line = np.poly1d(z)
                x_trend = np.linspace(min(rewards), max(rewards), 100)
                axes[3].plot(x_trend, trend_line(x_trend), 
                           '--', color=self.colors['warning'], linewidth=2)
                
                # 计算相关系数
                correlation = np.corrcoef(rewards, returns)[0, 1]
                axes[3].text(0.02, 0.98, f'相关系数: {correlation:.3f}',
                            transform=axes[3].transAxes,
                            verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
            
            axes[3].set_title('奖励-收益相关性', fontsize=14, fontweight='bold')
            axes[3].set_xlabel('累积奖励')
            axes[3].set_ylabel('收益率 (%)')
            axes[3].grid(True, alpha=0.3)
        
        # 5. 性能一致性分析
        if show_individual and len(episode_data) > 1:
            # 计算滚动平均
            window_size = max(3, len(episode_data) // 4)
            rolling_returns = pd.Series(returns).rolling(window=window_size, min_periods=1).mean()
            
            axes[4].plot(episodes, returns, 'o-', color=self.colors['primary'], 
                        alpha=0.6, label='单个Episode', markersize=4)
            axes[4].plot(episodes, rolling_returns, color=self.colors['success'], 
                        linewidth=3, label=f'滚动平均({window_size})')
            axes[4].axhline(y=0, color='black', linestyle='--', alpha=0.5)
            axes[4].set_title('性能一致性', fontsize=14, fontweight='bold')
            axes[4].set_xlabel('Episode')
            axes[4].set_ylabel('收益率 (%)')
            axes[4].legend()
            axes[4].grid(True, alpha=0.3)
        
        # 6. 统计摘要
        axes[5].axis('off')
        
        # 计算统计指标
        avg_reward = np.mean(rewards)
        avg_return = np.mean(returns)
        std_return = np.std(returns)
        win_rate = len([r for r in returns if r > 0]) / len(returns)
        best_episode = episodes[np.argmax(returns)] if returns else 0
        worst_episode = episodes[np.argmin(returns)] if returns else 0
        sharpe_ratio = (avg_return / std_return) if std_return > 0 else 0
        
        stats_text = (
            f'评估统计摘要\n\n'
            f'测试Episodes: {len(episode_data)}\n'
            f'平均奖励: {avg_reward:.3f}\n'
            f'平均收益: {avg_return:.2f}%\n'
            f'收益标准差: {std_return:.2f}%\n'
            f'胜率: {win_rate:.2%}\n'
            f'夏普比率: {sharpe_ratio:.3f}\n\n'
            f'最佳Episode: #{best_episode} ({max(returns):.2f}%)\n'
            f'最差Episode: #{worst_episode} ({min(returns):.2f}%)\n'
            f'平均步数: {np.mean(steps):.0f}\n'
            f'最终平均价值: ${np.mean(final_values):,.0f}'
        )
        
        axes[5].text(0.1, 0.9, stats_text,
                    transform=axes[5].transAxes,
                    fontsize=11,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        return self.save_figure(fig, 'episode_performance', 'evaluation')
    
    def plot_trading_signals(self, 
                            prices: List[float],
                            actions: List[float],
                            portfolio_values: List[float],
                            timestamps: Optional[List] = None) -> List[str]:
        """
        绘制交易信号和价格走势
        
        Args:
            prices: 价格序列
            actions: 交易动作序列
            portfolio_values: 投资组合价值序列
            timestamps: 时间戳 (可选)
            
        Returns:
            List[str]: 保存的文件路径
        """
        fig, axes = self.create_figure(figsize=(16, 12), subplots=(3, 1))
        
        # 创建时间轴
        if timestamps is None:
            timestamps = np.arange(len(prices))
        
        # 1. 价格走势和交易信号
        axes[0].plot(timestamps, prices, color=self.colors['primary'], 
                    linewidth=1.5, label='价格')
        
        # 标记买卖信号
        buy_signals = []
        sell_signals = []
        buy_times = []
        sell_times = []
        
        for i, action in enumerate(actions):
            if action > 0.5:  # 买入信号
                buy_signals.append(prices[i])
                buy_times.append(timestamps[i])
            elif action < -0.5:  # 卖出信号
                sell_signals.append(prices[i])
                sell_times.append(timestamps[i])
        
        if buy_signals:
            axes[0].scatter(buy_times, buy_signals, color=self.colors['profit'], 
                           s=50, marker='^', label='买入信号', alpha=0.8, zorder=5)
        if sell_signals:
            axes[0].scatter(sell_times, sell_signals, color=self.colors['loss'], 
                           s=50, marker='v', label='卖出信号', alpha=0.8, zorder=5)
        
        axes[0].set_title('价格走势与交易信号', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('价格')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 2. 动作强度热力图
        axes[1].plot(timestamps, actions, color=self.colors['secondary'], 
                    linewidth=2, label='交易动作')
        axes[1].fill_between(timestamps, actions, 0,
                           where=np.array(actions) >= 0,
                           color=self.colors['profit'], alpha=0.3, label='做多')
        axes[1].fill_between(timestamps, actions, 0,
                           where=np.array(actions) < 0,
                           color=self.colors['loss'], alpha=0.3, label='做空')
        axes[1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[1].set_title('交易动作强度', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('动作强度')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # 3. 投资组合价值演化
        axes[2].plot(timestamps, portfolio_values, color=self.colors['success'], 
                    linewidth=2, label='投资组合价值')
        
        # 添加初始资金线
        initial_value = portfolio_values[0] if portfolio_values else 10000
        axes[2].axhline(y=initial_value, color='black', linestyle='--', 
                       alpha=0.5, label=f'初始资金: ${initial_value:,.0f}')
        
        # 填充盈亏区域
        axes[2].fill_between(timestamps, portfolio_values, initial_value,
                           where=np.array(portfolio_values) >= initial_value,
                           color=self.colors['profit'], alpha=0.3)
        axes[2].fill_between(timestamps, portfolio_values, initial_value,
                           where=np.array(portfolio_values) < initial_value,
                           color=self.colors['loss'], alpha=0.3)
        
        axes[2].set_title('投资组合价值演化', fontsize=14, fontweight='bold')
        axes[2].set_xlabel('时间/步数')
        axes[2].set_ylabel('投资组合价值 ($)')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        # 添加性能统计
        if portfolio_values:
            final_value = portfolio_values[-1]
            total_return = (final_value - initial_value) / initial_value * 100
            max_value = max(portfolio_values)
            min_value = min(portfolio_values)
            
            stats_text = (
                f'交易统计:\n'
                f'买入信号: {len(buy_signals)}次\n'
                f'卖出信号: {len(sell_signals)}次\n'
                f'总收益: {total_return:.2f}%\n'
                f'最高价值: ${max_value:,.0f}\n'
                f'最低价值: ${min_value:,.0f}'
            )
            
            axes[2].text(0.02, 0.98, stats_text,
                        transform=axes[2].transAxes,
                        verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        return self.save_figure(fig, 'trading_signals', 'evaluation')
    
    def plot_risk_return_analysis(self, 
                                 episode_data: List[Dict[str, Any]],
                                 benchmark_return: float = 0.0,
                                 risk_free_rate: float = 0.02) -> List[str]:
        """
        绘制风险收益分析
        
        Args:
            episode_data: Episode数据
            benchmark_return: 基准收益率
            risk_free_rate: 无风险利率
            
        Returns:
            List[str]: 保存的文件路径
        """
        fig, axes = self.create_figure(figsize=(15, 10), subplots=(2, 2))
        
        # 提取数据
        returns = [d.get('return', 0) / 100 for d in episode_data]  # 转换为小数
        volatilities = [d.get('volatility', 0) for d in episode_data]
        max_drawdowns = [d.get('max_drawdown', 0) for d in episode_data]
        sharpe_ratios = [(r - risk_free_rate) / v if v > 0 else 0 
                        for r, v in zip(returns, volatilities)]
        
        # 如果没有波动率数据，计算简单波动率
        if not any(volatilities):
            volatilities = [abs(r) * 0.1 for r in returns]  # 简单估计
        
        # 1. 风险-收益散点图
        colors = [self.colors['profit'] if r >= 0 else self.colors['loss'] for r in returns]
        scatter = axes[0].scatter(volatilities, returns, c=colors, 
                                s=60, alpha=0.7, edgecolors='black', linewidth=0.5)
        
        # 添加效率前沿（简化版）
        if len(returns) > 3:
            # 找出帕累托前沿点
            efficient_points_x = []
            efficient_points_y = []
            for i, (vol, ret) in enumerate(zip(volatilities, returns)):
                is_efficient = True
                for j, (vol2, ret2) in enumerate(zip(volatilities, returns)):
                    if i != j and vol2 <= vol and ret2 >= ret and (vol2 < vol or ret2 > ret):
                        is_efficient = False
                        break
                if is_efficient:
                    efficient_points_x.append(vol)
                    efficient_points_y.append(ret)
            
            if efficient_points_x:
                # 排序并连线
                sorted_indices = np.argsort(efficient_points_x)
                sorted_x = [efficient_points_x[i] for i in sorted_indices]
                sorted_y = [efficient_points_y[i] for i in sorted_indices]
                axes[0].plot(sorted_x, sorted_y, '--', color=self.colors['warning'], 
                           linewidth=2, label='效率前沿', alpha=0.8)
        
        # 添加基准线
        if benchmark_return != 0:
            axes[0].axhline(y=benchmark_return, color='gray', 
                          linestyle=':', linewidth=2, label=f'基准收益: {benchmark_return:.2%}')
        
        axes[0].set_title('风险-收益分析', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('波动率 (风险)')
        axes[0].set_ylabel('收益率')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 2. 夏普比率分布
        episodes = list(range(1, len(sharpe_ratios) + 1))
        colors = [self.colors['profit'] if s >= 0 else self.colors['loss'] for s in sharpe_ratios]
        axes[1].bar(episodes, sharpe_ratios, color=colors, alpha=0.7, 
                   edgecolor='black', linewidth=0.5)
        axes[1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[1].set_title('夏普比率分布', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Episode')
        axes[1].set_ylabel('夏普比率')
        axes[1].grid(True, alpha=0.3)
        
        # 3. 最大回撤分析
        axes[2].bar(episodes, max_drawdowns, color=self.colors['loss'], 
                   alpha=0.7, edgecolor='black', linewidth=0.5)
        axes[2].set_title('最大回撤分析', fontsize=14, fontweight='bold')
        axes[2].set_xlabel('Episode')
        axes[2].set_ylabel('最大回撤 (%)')
        axes[2].grid(True, alpha=0.3)
        axes[2].invert_yaxis()  # 回撤向下显示
        
        # 4. 风险调整后收益 (Calmar比率)
        calmar_ratios = [abs(r) / abs(dd) if dd != 0 else 0 
                        for r, dd in zip(returns, max_drawdowns)]
        
        axes[3].bar(episodes, calmar_ratios, color=self.colors['info'], 
                   alpha=0.7, edgecolor='black', linewidth=0.5)
        axes[3].set_title('Calmar比率 (收益/最大回撤)', fontsize=14, fontweight='bold')
        axes[3].set_xlabel('Episode')
        axes[3].set_ylabel('Calmar比率')
        axes[3].grid(True, alpha=0.3)
        
        # 添加统计摘要
        avg_return = np.mean(returns) * 100
        avg_volatility = np.mean(volatilities)
        avg_sharpe = np.mean(sharpe_ratios)
        avg_drawdown = np.mean(max_drawdowns)
        avg_calmar = np.mean(calmar_ratios)
        
        stats_text = (
            f'风险收益统计:\n\n'
            f'平均收益: {avg_return:.2f}%\n'
            f'平均波动率: {avg_volatility:.3f}\n'
            f'平均夏普比率: {avg_sharpe:.3f}\n'
            f'平均最大回撤: {avg_drawdown:.2f}%\n'
            f'平均Calmar比率: {avg_calmar:.3f}\n\n'
            f'最佳夏普比率: {max(sharpe_ratios):.3f}\n'
            f'最大回撤: {max(max_drawdowns):.2f}%'
        )
        
        axes[3].text(0.02, 0.98, stats_text,
                    transform=axes[3].transAxes,
                    verticalalignment='top',
                    fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        
        return self.save_figure(fig, 'risk_return_analysis', 'evaluation')
    
    def create_evaluation_report(self, 
                                episode_data: List[Dict[str, Any]],
                                model_info: Dict[str, Any],
                                test_period: str = "") -> List[str]:
        """
        创建完整的评估报告
        
        Args:
            episode_data: Episode数据
            model_info: 模型信息
            test_period: 测试期间描述
            
        Returns:
            List[str]: 保存的文件路径
        """
        fig, axes = self.create_figure(figsize=(20, 16), subplots=(4, 3))
        
        # 提取基础数据
        episodes = [d.get('episode', i+1) for i, d in enumerate(episode_data)]
        returns = [d.get('return', 0) for d in episode_data]
        rewards = [d.get('reward', 0) for d in episode_data]
        steps = [d.get('steps', 0) for d in episode_data]
        
        # 1. 收益率分布
        axes[0].hist(returns, bins=20, alpha=0.7, color=self.colors['primary'], 
                    edgecolor='black')
        axes[0].axvline(np.mean(returns), color='red', linestyle='--', linewidth=2)
        axes[0].set_title('收益率分布', fontweight='bold')
        axes[0].set_xlabel('收益率 (%)')
        
        # 2. 收益率时间序列
        axes[1].plot(episodes, returns, 'o-', color=self.colors['success'], 
                    linewidth=2, markersize=4)
        axes[1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[1].set_title('收益率时间序列', fontweight='bold')
        axes[1].set_xlabel('Episode')
        axes[1].set_ylabel('收益率 (%)')
        axes[1].grid(True, alpha=0.3)
        
        # 3. 奖励分布
        axes[2].hist(rewards, bins=20, alpha=0.7, color=self.colors['secondary'], 
                    edgecolor='black')
        axes[2].axvline(np.mean(rewards), color='red', linestyle='--', linewidth=2)
        axes[2].set_title('奖励分布', fontweight='bold')
        axes[2].set_xlabel('累积奖励')
        
        # 4. 步数统计
        axes[3].bar(episodes, steps, color=self.colors['info'], alpha=0.7)
        axes[3].set_title('Episode步数', fontweight='bold')
        axes[3].set_xlabel('Episode')
        axes[3].set_ylabel('步数')
        
        # 5. 累积收益曲线
        cumulative_returns = np.cumsum(returns)
        axes[4].plot(episodes, cumulative_returns, color=self.colors['warning'], 
                    linewidth=3)
        axes[4].set_title('累积收益曲线', fontweight='bold')
        axes[4].set_xlabel('Episode')
        axes[4].set_ylabel('累积收益率 (%)')
        axes[4].grid(True, alpha=0.3)
        
        # 6. 胜率统计
        win_episodes = len([r for r in returns if r > 0])
        lose_episodes = len([r for r in returns if r < 0])
        draw_episodes = len([r for r in returns if r == 0])
        
        win_data = [win_episodes, lose_episodes, draw_episodes]
        win_labels = ['盈利', '亏损', '平局']
        win_colors = [self.colors['profit'], self.colors['loss'], self.colors['neutral']]
        
        axes[5].pie(win_data, labels=win_labels, colors=win_colors, autopct='%1.1f%%')
        axes[5].set_title('胜率统计', fontweight='bold')
        
        # 7-12. 模型信息和统计摘要
        for idx in range(6, 12):
            axes[idx].axis('off')
        
        # 模型信息
        model_text = (
            f'模型信息\n\n'
            f'模型类型: {model_info.get("model_type", "未知")}\n'
            f'算法: {model_info.get("algorithm", "未知")}\n'
            f'奖励函数: {model_info.get("reward_type", "未知")}\n'
            f'训练数据: {model_info.get("training_symbol", "未知")}\n'
            f'测试期间: {test_period}\n'
            f'评估时间: {datetime.now().strftime("%Y-%m-%d %H:%M")}'
        )
        
        axes[6].text(0.1, 0.9, model_text,
                    transform=axes[6].transAxes,
                    fontsize=11,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))
        
        # 性能统计
        avg_return = np.mean(returns)
        std_return = np.std(returns)
        max_return = max(returns) if returns else 0
        min_return = min(returns) if returns else 0
        win_rate = win_episodes / len(returns) if returns else 0
        profit_factor = abs(sum([r for r in returns if r > 0])) / abs(sum([r for r in returns if r < 0])) if any(r < 0 for r in returns) else float('inf')
        
        perf_text = (
            f'性能统计\n\n'
            f'总Episodes: {len(episode_data)}\n'
            f'平均收益: {avg_return:.2f}%\n'
            f'收益标准差: {std_return:.2f}%\n'
            f'最大收益: {max_return:.2f}%\n'
            f'最大亏损: {min_return:.2f}%\n'
            f'胜率: {win_rate:.2%}\n'
            f'盈亏比: {profit_factor:.2f}\n'
            f'累积收益: {cumulative_returns[-1]:.2f}%'
        )
        
        axes[7].text(0.1, 0.9, perf_text,
                    transform=axes[7].transAxes,
                    fontsize=11,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        
        # 风险指标
        sharpe_ratio = avg_return / std_return if std_return > 0 else 0
        downside_returns = [r for r in returns if r < 0]
        downside_deviation = np.std(downside_returns) if downside_returns else 0
        sortino_ratio = avg_return / downside_deviation if downside_deviation > 0 else float('inf')
        
        risk_text = (
            f'风险指标\n\n'
            f'夏普比率: {sharpe_ratio:.3f}\n'
            f'索提诺比率: {sortino_ratio:.3f}\n'
            f'下行标准差: {downside_deviation:.2f}%\n'
            f'最大连续亏损: {self._calculate_max_consecutive_losses(returns)}\n'
            f'波动率: {std_return:.2f}%\n'
            f'VaR (95%): {np.percentile(returns, 5):.2f}%\n'
            f'CVaR (95%): {np.mean([r for r in returns if r <= np.percentile(returns, 5)]):.2f}%'
        )
        
        axes[8].text(0.1, 0.9, risk_text,
                    transform=axes[8].transAxes,
                    fontsize=11,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        return self.save_figure(fig, 'evaluation_report', 'evaluation')
    
    def _calculate_max_consecutive_losses(self, returns: List[float]) -> int:
        """计算最大连续亏损次数"""
        max_consecutive = 0
        current_consecutive = 0
        
        for ret in returns:
            if ret < 0:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
                
        return max_consecutive