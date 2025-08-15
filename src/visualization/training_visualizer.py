"""
训练过程可视化器

专门用于可视化强化学习训练过程中的各项指标和数据。
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import logging
from scipy import stats
from scipy.ndimage import gaussian_filter1d

from .base_visualizer import BaseVisualizer


class TrainingVisualizer(BaseVisualizer):
    """
    训练过程可视化器
    
    专门处理训练相关的数据可视化，包括：
    - 奖励曲线
    - 投资组合价值变化
    - 交易动作分析
    - 学习进度监控
    - 性能指标趋势
    """
    
    def __init__(self, **kwargs):
        """初始化训练可视化器"""
        super().__init__(**kwargs)
        # 使用统一日志系统
        from ..utils.logger import get_logger
        self.logger = get_logger('TrainingVisualizer')
    
    def plot_reward_curves(self, 
                          episode_rewards: List[float],
                          smoothing_window: int = 100,
                          show_trend: bool = True) -> List[str]:
        """
        绘制奖励曲线
        
        Args:
            episode_rewards: 每个episode的奖励
            smoothing_window: 平滑窗口大小
            show_trend: 是否显示趋势线
            
        Returns:
            List[str]: 保存的文件路径
        """
        fig, axes = self.create_figure(figsize=(15, 10), subplots=(2, 2))
        
        episodes = np.arange(len(episode_rewards))
        
        # 1. 原始奖励曲线
        axes[0].plot(episodes, episode_rewards, 
                    alpha=0.6, color=self.colors['primary'], linewidth=0.8)
        axes[0].set_title('原始奖励曲线', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Episode')
        axes[0].set_ylabel('奖励')
        axes[0].grid(True, alpha=0.3)
        
        # 添加零线
        axes[0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        # 2. 平滑奖励曲线
        if len(episode_rewards) > smoothing_window:
            smoothed_rewards = pd.Series(episode_rewards).rolling(
                window=smoothing_window, min_periods=1).mean()
            
            axes[1].plot(episodes, episode_rewards, 
                        alpha=0.3, color='lightgray', linewidth=0.5, label='原始数据')
            axes[1].plot(episodes, smoothed_rewards, 
                        color=self.colors['success'], linewidth=2, label=f'平滑({smoothing_window} episodes)')
            
            if show_trend:
                # 添加趋势线
                z = np.polyfit(episodes, smoothed_rewards, 1)
                trend_line = np.poly1d(z)
                axes[1].plot(episodes, trend_line(episodes), 
                           '--', color=self.colors['warning'], linewidth=2, label='趋势线')
            
            axes[1].set_title('平滑奖励曲线', fontsize=14, fontweight='bold')
            axes[1].set_xlabel('Episode')
            axes[1].set_ylabel('平滑奖励')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        
        # 3. 奖励分布直方图
        axes[2].hist(episode_rewards, bins=50, alpha=0.7, 
                    color=self.colors['info'], edgecolor='black')
        axes[2].axvline(np.mean(episode_rewards), color=self.colors['warning'], 
                       linestyle='--', linewidth=2, label=f'均值: {np.mean(episode_rewards):.3f}')
        axes[2].axvline(np.median(episode_rewards), color=self.colors['success'], 
                       linestyle='--', linewidth=2, label=f'中位数: {np.median(episode_rewards):.3f}')
        axes[2].set_title('奖励分布', fontsize=14, fontweight='bold')
        axes[2].set_xlabel('奖励值')
        axes[2].set_ylabel('频次')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        # 4. 累积奖励
        cumulative_rewards = np.cumsum(episode_rewards)
        axes[3].plot(episodes, cumulative_rewards, 
                    color=self.colors['secondary'], linewidth=2)
        axes[3].set_title('累积奖励', fontsize=14, fontweight='bold')
        axes[3].set_xlabel('Episode')
        axes[3].set_ylabel('累积奖励')
        axes[3].grid(True, alpha=0.3)
        
        # 添加统计信息文本框
        stats_text = (
            f'总Episodes: {len(episode_rewards)}\n'
            f'平均奖励: {np.mean(episode_rewards):.3f}\n'
            f'标准差: {np.std(episode_rewards):.3f}\n'
            f'最大值: {np.max(episode_rewards):.3f}\n'
            f'最小值: {np.min(episode_rewards):.3f}\n'
            f'最终累积: {cumulative_rewards[-1]:.3f}'
        )
        axes[3].text(0.02, 0.98, stats_text,
                    transform=axes[3].transAxes,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        return self.save_figure(fig, 'training_reward_analysis', 'training')
    
    def plot_portfolio_evolution(self, 
                                portfolio_values: List[float],
                                initial_balance: float = 10000.0,
                                benchmark_return: float = 0.0) -> List[str]:
        """
        绘制投资组合价值演化
        
        Args:
            portfolio_values: 投资组合价值列表
            initial_balance: 初始资金
            benchmark_return: 基准收益率
            
        Returns:
            List[str]: 保存的文件路径
        """
        fig, axes = self.create_figure(figsize=(15, 10), subplots=(2, 2))
        
        episodes = np.arange(len(portfolio_values))
        returns = [(v - initial_balance) / initial_balance * 100 for v in portfolio_values]
        
        # 1. 投资组合价值变化
        axes[0].plot(episodes, portfolio_values, 
                    color=self.colors['primary'], linewidth=2)
        axes[0].axhline(y=initial_balance, color='black', 
                       linestyle='--', alpha=0.5, label=f'初始资金: ${initial_balance:,.0f}')
        axes[0].fill_between(episodes, portfolio_values, initial_balance,
                           where=np.array(portfolio_values) >= initial_balance,
                           color=self.colors['profit'], alpha=0.3, interpolate=True)
        axes[0].fill_between(episodes, portfolio_values, initial_balance,
                           where=np.array(portfolio_values) < initial_balance,
                           color=self.colors['loss'], alpha=0.3, interpolate=True)
        axes[0].set_title('投资组合价值演化', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Episode')
        axes[0].set_ylabel('投资组合价值 ($)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 2. 收益率曲线
        axes[1].plot(episodes, returns, color=self.colors['success'], linewidth=2)
        axes[1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        if benchmark_return != 0:
            benchmark_line = [benchmark_return * 100] * len(episodes)
            axes[1].plot(episodes, benchmark_line, 
                        color=self.colors['warning'], linestyle='--', 
                        linewidth=2, label=f'基准: {benchmark_return:.2%}')
            axes[1].legend()
        axes[1].set_title('收益率曲线', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Episode')
        axes[1].set_ylabel('收益率 (%)')
        axes[1].grid(True, alpha=0.3)
        
        # 3. 回撤分析
        peak_values = np.maximum.accumulate(portfolio_values)
        drawdowns = [(peak - current) / peak * 100 
                    for peak, current in zip(peak_values, portfolio_values)]
        
        axes[2].fill_between(episodes, 0, drawdowns, 
                           color=self.colors['loss'], alpha=0.7)
        axes[2].set_title('回撤分析', fontsize=14, fontweight='bold')
        axes[2].set_xlabel('Episode')
        axes[2].set_ylabel('回撤 (%)')
        axes[2].grid(True, alpha=0.3)
        axes[2].invert_yaxis()  # 回撤向下显示
        
        # 4. 收益分布
        axes[3].hist(returns, bins=30, alpha=0.7, 
                    color=self.colors['info'], edgecolor='black')
        axes[3].axvline(np.mean(returns), color=self.colors['warning'], 
                       linestyle='--', linewidth=2, label=f'均值: {np.mean(returns):.2f}%')
        axes[3].set_title('收益分布', fontsize=14, fontweight='bold')
        axes[3].set_xlabel('收益率 (%)')
        axes[3].set_ylabel('频次')
        axes[3].legend()
        axes[3].grid(True, alpha=0.3)
        
        # 添加性能统计
        max_drawdown = max(drawdowns) if drawdowns else 0
        final_return = returns[-1] if returns else 0
        volatility = np.std(returns) if len(returns) > 1 else 0
        sharpe_ratio = (np.mean(returns) / volatility) if volatility > 0 else 0
        
        stats_text = (
            f'最终收益: {final_return:.2f}%\n'
            f'最大回撤: {max_drawdown:.2f}%\n'
            f'波动率: {volatility:.2f}%\n'
            f'夏普比率: {sharpe_ratio:.3f}\n'
            f'胜率: {len([r for r in returns if r > 0]) / len(returns):.2%}'
        )
        axes[3].text(0.02, 0.98, stats_text,
                    transform=axes[3].transAxes,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        return self.save_figure(fig, 'portfolio_evolution', 'training')
    
    def plot_action_analysis(self, 
                           actions_history: List[float],
                           portfolio_values: List[float]) -> List[str]:
        """
        绘制交易动作分析
        
        Args:
            actions_history: 交易动作历史
            portfolio_values: 投资组合价值
            
        Returns:
            List[str]: 保存的文件路径
        """
        fig, axes = self.create_figure(figsize=(15, 12), subplots=(3, 2))
        
        episodes = np.arange(len(actions_history))
        
        # 1. 动作时间序列
        axes[0].plot(episodes, actions_history, 
                    color=self.colors['primary'], linewidth=1.5, alpha=0.8)
        axes[0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[0].fill_between(episodes, actions_history, 0,
                           where=np.array(actions_history) >= 0,
                           color=self.colors['profit'], alpha=0.3, label='买入/持有')
        axes[0].fill_between(episodes, actions_history, 0,
                           where=np.array(actions_history) < 0,
                           color=self.colors['loss'], alpha=0.3, label='卖出/空仓')
        axes[0].set_title('交易动作时间序列', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Episode')
        axes[0].set_ylabel('动作强度')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 2. 动作分布
        axes[1].hist(actions_history, bins=50, alpha=0.7, 
                    color=self.colors['info'], edgecolor='black')
        axes[1].axvline(np.mean(actions_history), color=self.colors['warning'], 
                       linestyle='--', linewidth=2, label=f'均值: {np.mean(actions_history):.3f}')
        axes[1].set_title('动作分布', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('动作值')
        axes[1].set_ylabel('频次')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # 3. 动作-收益相关性 (如果数据长度匹配)
        if len(actions_history) == len(portfolio_values):
            returns = np.diff(portfolio_values)
            actions_subset = actions_history[1:]  # 对齐长度
            
            if len(returns) > 1 and len(actions_subset) > 1:
                axes[2].scatter(actions_subset, returns, 
                              alpha=0.6, color=self.colors['secondary'])
                
                # 添加趋势线
                if len(actions_subset) > 10:
                    z = np.polyfit(actions_subset, returns, 1)
                    trend_line = np.poly1d(z)
                    x_trend = np.linspace(min(actions_subset), max(actions_subset), 100)
                    axes[2].plot(x_trend, trend_line(x_trend), 
                               '--', color=self.colors['warning'], linewidth=2)
                    
                    # 计算相关系数
                    correlation = np.corrcoef(actions_subset, returns)[0, 1]
                    axes[2].text(0.02, 0.98, f'相关系数: {correlation:.3f}',
                                transform=axes[2].transAxes,
                                verticalalignment='top',
                                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
                
                axes[2].set_title('动作-收益相关性', fontsize=14, fontweight='bold')
                axes[2].set_xlabel('动作值')
                axes[2].set_ylabel('投资组合价值变化')
                axes[2].grid(True, alpha=0.3)
        
        # 4. 动作变化率
        action_changes = np.diff(actions_history)
        axes[3].plot(episodes[1:], action_changes, 
                    color=self.colors['success'], linewidth=1.5, alpha=0.8)
        axes[3].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[3].set_title('动作变化率', fontsize=14, fontweight='bold')
        axes[3].set_xlabel('Episode')
        axes[3].set_ylabel('动作变化')
        axes[3].grid(True, alpha=0.3)
        
        # 5. 动作强度热力图 (滑动窗口)
        if len(actions_history) > 100:
            window_size = 50
            action_intensity = []
            for i in range(window_size, len(actions_history)):
                window_actions = actions_history[i-window_size:i]
                intensity = np.std(window_actions)  # 使用标准差作为强度指标
                action_intensity.append(intensity)
            
            axes[4].plot(episodes[window_size:], action_intensity, 
                        color=self.colors['warning'], linewidth=2)
            axes[4].set_title('动作强度演化', fontsize=14, fontweight='bold')
            axes[4].set_xlabel('Episode')
            axes[4].set_ylabel('动作强度 (标准差)')
            axes[4].grid(True, alpha=0.3)
        
        # 6. 动作统计摘要
        axes[5].axis('off')
        
        # 计算统计指标
        long_actions = [a for a in actions_history if a > 0.1]
        short_actions = [a for a in actions_history if a < -0.1]
        neutral_actions = [a for a in actions_history if abs(a) <= 0.1]
        
        stats_text = (
            f'动作统计摘要:\n\n'
            f'总动作数: {len(actions_history)}\n'
            f'做多次数: {len(long_actions)} ({len(long_actions)/len(actions_history):.1%})\n'
            f'做空次数: {len(short_actions)} ({len(short_actions)/len(actions_history):.1%})\n'
            f'中性次数: {len(neutral_actions)} ({len(neutral_actions)/len(actions_history):.1%})\n\n'
            f'平均动作: {np.mean(actions_history):.3f}\n'
            f'动作标准差: {np.std(actions_history):.3f}\n'
            f'最大做多: {max(actions_history):.3f}\n'
            f'最大做空: {min(actions_history):.3f}\n\n'
            f'动作变化频率: {np.mean(np.abs(np.diff(actions_history))):.3f}'
        )
        
        axes[5].text(0.1, 0.9, stats_text,
                    transform=axes[5].transAxes,
                    fontsize=11,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        
        return self.save_figure(fig, 'action_analysis', 'training')
    
    def plot_learning_progress(self, 
                              training_metrics: Dict[str, List[float]],
                              smoothing_factor: float = 0.9) -> List[str]:
        """
        绘制学习进度监控
        
        Args:
            training_metrics: 训练指标字典 (包含 loss, entropy, value_loss等)
            smoothing_factor: 指数平滑因子
            
        Returns:
            List[str]: 保存的文件路径
        """
        if not training_metrics:
            self.logger.warning("没有提供训练指标数据")
            return []
        
        # 计算子图布局
        n_metrics = len(training_metrics)
        cols = min(3, n_metrics)
        rows = (n_metrics + cols - 1) // cols
        
        fig, axes = self.create_figure(figsize=(5*cols, 4*rows), subplots=(rows, cols))
        
        if n_metrics == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes if isinstance(axes, (list, np.ndarray)) else [axes]
        else:
            axes = axes.flatten()
        
        for idx, (metric_name, values) in enumerate(training_metrics.items()):
            if idx >= len(axes):
                break
                
            ax = axes[idx]
            steps = np.arange(len(values))
            
            # 原始数据
            ax.plot(steps, values, alpha=0.3, color='lightgray', 
                   linewidth=0.8, label='原始数据')
            
            # 指数平滑
            if len(values) > 1:
                smoothed = []
                smoothed.append(values[0])
                for v in values[1:]:
                    smoothed.append(smoothing_factor * smoothed[-1] + (1 - smoothing_factor) * v)
                
                ax.plot(steps, smoothed, color=self.colors['primary'], 
                       linewidth=2, label='平滑数据')
            
            ax.set_title(f'{metric_name}', fontsize=12, fontweight='bold')
            ax.set_xlabel('Step')
            ax.set_ylabel(metric_name)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # 添加趋势指示
            if len(values) > 10:
                recent_trend = np.mean(values[-10:]) - np.mean(values[-20:-10]) if len(values) >= 20 else 0
                trend_color = self.colors['profit'] if recent_trend < 0 else self.colors['loss']  # loss下降是好的
                ax.text(0.02, 0.98, f'趋势: {"↓" if recent_trend < 0 else "↑"}',
                       transform=ax.transAxes,
                       verticalalignment='top',
                       color=trend_color,
                       fontweight='bold',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 隐藏多余的子图
        for idx in range(n_metrics, len(axes)):
            axes[idx].set_visible(False)
        
        return self.save_figure(fig, 'learning_progress', 'training')
    
    def create_training_dashboard(self,
                                 episode_rewards: List[float],
                                 portfolio_values: List[float],
                                 actions_history: List[float],
                                 training_metrics: Optional[Dict[str, List[float]]] = None,
                                 initial_balance: float = 10000.0) -> List[str]:
        """
        创建训练仪表板 - 综合所有关键指标
        
        Args:
            episode_rewards: 每个episode的奖励
            portfolio_values: 投资组合价值
            actions_history: 交易动作历史
            training_metrics: 训练指标 (可选)
            initial_balance: 初始资金
            
        Returns:
            List[str]: 保存的文件路径
        """
        fig, axes = self.create_figure(figsize=(20, 16), subplots=(4, 3))
        
        episodes = np.arange(len(episode_rewards))
        returns = [(v - initial_balance) / initial_balance * 100 for v in portfolio_values]
        
        # 1. 奖励曲线 + 平滑
        smoothed_rewards = pd.Series(episode_rewards).rolling(window=50, min_periods=1).mean()
        axes[0].plot(episodes, episode_rewards, alpha=0.3, color='lightgray', linewidth=0.5)
        axes[0].plot(episodes, smoothed_rewards, color=self.colors['primary'], linewidth=2)
        axes[0].set_title('奖励进度', fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        
        # 2. 投资组合价值
        axes[1].plot(episodes, portfolio_values, color=self.colors['success'], linewidth=2)
        axes[1].axhline(y=initial_balance, color='black', linestyle='--', alpha=0.5)
        axes[1].set_title('投资组合价值', fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        # 3. 收益率
        axes[2].plot(episodes, returns, color=self.colors['info'], linewidth=2)
        axes[2].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[2].set_title('收益率 (%)', fontweight='bold')
        axes[2].grid(True, alpha=0.3)
        
        # 4. 交易动作
        axes[3].plot(episodes, actions_history, color=self.colors['secondary'], linewidth=1.5, alpha=0.8)
        axes[3].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[3].set_title('交易动作', fontweight='bold')
        axes[3].grid(True, alpha=0.3)
        
        # 5. 奖励分布
        axes[4].hist(episode_rewards, bins=30, alpha=0.7, color=self.colors['primary'])
        axes[4].axvline(np.mean(episode_rewards), color='red', linestyle='--')
        axes[4].set_title('奖励分布', fontweight='bold')
        
        # 6. 动作分布
        axes[5].hist(actions_history, bins=30, alpha=0.7, color=self.colors['secondary'])
        axes[5].axvline(np.mean(actions_history), color='red', linestyle='--')
        axes[5].set_title('动作分布', fontweight='bold')
        
        # 7. 回撤分析
        peak_values = np.maximum.accumulate(portfolio_values)
        drawdowns = [(peak - current) / peak * 100 for peak, current in zip(peak_values, portfolio_values)]
        axes[6].fill_between(episodes, 0, drawdowns, color=self.colors['loss'], alpha=0.7)
        axes[6].set_title('回撤分析', fontweight='bold')
        axes[6].invert_yaxis()
        
        # 8. 累积奖励
        cumulative_rewards = np.cumsum(episode_rewards)
        axes[7].plot(episodes, cumulative_rewards, color=self.colors['warning'], linewidth=2)
        axes[7].set_title('累积奖励', fontweight='bold')
        axes[7].grid(True, alpha=0.3)
        
        # 9. 学习稳定性 (奖励方差)
        if len(episode_rewards) > 50:
            variance_window = 50
            reward_variance = []
            for i in range(variance_window, len(episode_rewards)):
                window_rewards = episode_rewards[i-variance_window:i]
                variance = np.var(window_rewards)
                reward_variance.append(variance)
            
            axes[8].plot(episodes[variance_window:], reward_variance, 
                        color=self.colors['info'], linewidth=2)
            axes[8].set_title('学习稳定性', fontweight='bold')
            axes[8].grid(True, alpha=0.3)
        
        # 10-12. 性能统计表格
        for idx in [9, 10, 11]:
            axes[idx].axis('off')
        
        # 计算关键统计信息
        final_return = returns[-1] if returns else 0
        max_drawdown = max(drawdowns) if drawdowns else 0
        volatility = np.std(returns) if len(returns) > 1 else 0
        sharpe_ratio = (np.mean(returns) / volatility) if volatility > 0 else 0
        win_rate = len([r for r in returns if r > 0]) / len(returns) if returns else 0
        
        stats_text = (
            f'训练统计摘要\n\n'
            f'总Episodes: {len(episode_rewards)}\n'
            f'平均奖励: {np.mean(episode_rewards):.3f}\n'
            f'最终收益: {final_return:.2f}%\n'
            f'最大回撤: {max_drawdown:.2f}%\n'
            f'夏普比率: {sharpe_ratio:.3f}\n'
            f'胜率: {win_rate:.2%}\n'
            f'波动率: {volatility:.2f}%\n\n'
            f'动作统计:\n'
            f'平均动作: {np.mean(actions_history):.3f}\n'
            f'动作标准差: {np.std(actions_history):.3f}\n'
        )
        
        axes[9].text(0.1, 0.9, stats_text,
                    transform=axes[9].transAxes,
                    fontsize=10,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        return self.save_figure(fig, 'training_dashboard', 'training')