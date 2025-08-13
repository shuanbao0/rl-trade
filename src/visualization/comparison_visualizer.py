"""
模型对比可视化器

专门用于对比多个模型、实验或时间段的性能表现。
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
import logging
from scipy import stats

from .base_visualizer import BaseVisualizer


class ComparisonVisualizer(BaseVisualizer):
    """
    模型对比可视化器
    
    专门处理多模型、多实验、多时间段的对比分析，包括：
    - 性能指标对比
    - 收益曲线对比
    - 风险分析对比
    - 统计显著性检验
    - 排名分析
    - 稳定性对比
    """
    
    def __init__(self, **kwargs):
        """初始化对比可视化器"""
        super().__init__(**kwargs)
        self.logger = logging.getLogger('ComparisonVisualizer')
        
        # 对比专用颜色方案
        self.comparison_colors = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
        ]
    
    def compare_performance_metrics(self, 
                                   models_data: Dict[str, Dict[str, Any]],
                                   metrics: List[str] = None) -> List[str]:
        """
        对比多个模型的性能指标
        
        Args:
            models_data: 模型数据字典
                格式: {'Model_A': {'returns': [...], 'sharpe': 1.2, ...}, ...}
            metrics: 要对比的指标列表
            
        Returns:
            List[str]: 保存的文件路径
        """
        if not models_data:
            self.logger.warning("没有提供模型数据")
            return []
        
        if metrics is None:
            # 默认对比指标
            metrics = ['mean_return', 'std_return', 'sharpe_ratio', 'max_drawdown', 
                      'win_rate', 'calmar_ratio', 'sortino_ratio']
        
        fig, axes = self.create_figure(figsize=(16, 12), subplots=(3, 3))
        
        model_names = list(models_data.keys())
        n_models = len(model_names)
        
        # 计算每个模型的指标
        metrics_data = {}
        for metric in metrics:
            metrics_data[metric] = []
            
            for model_name in model_names:
                model_data = models_data[model_name]
                
                if metric in model_data:
                    # 直接使用提供的指标值
                    value = model_data[metric]
                else:
                    # 从基础数据计算指标
                    value = self._calculate_metric(model_data, metric)
                
                metrics_data[metric].append(value)
        
        # 1. 平均收益率对比
        if 'mean_return' in metrics_data:
            ax = axes[0]
            bars = ax.bar(model_names, metrics_data['mean_return'], 
                         color=self.comparison_colors[:n_models], alpha=0.7)
            ax.set_title('Average Return Comparison', fontsize=14, fontweight='bold')
            ax.set_ylabel('Return (%)')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)
            
            # 添加数值标签
            for bar, value in zip(bars, metrics_data['mean_return']):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.2f}%', ha='center', va='bottom')
        
        # 2. 夏普比率对比
        if 'sharpe_ratio' in metrics_data:
            ax = axes[1]
            bars = ax.bar(model_names, metrics_data['sharpe_ratio'],
                         color=self.comparison_colors[:n_models], alpha=0.7)
            ax.set_title('Sharpe Ratio Comparison', fontsize=14, fontweight='bold')
            ax.set_ylabel('Sharpe Ratio')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)
            ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Excellence Line (1.0)')
            ax.legend()
            
            for bar, value in zip(bars, metrics_data['sharpe_ratio']):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.3f}', ha='center', va='bottom')
        
        # 3. 最大回撤对比
        if 'max_drawdown' in metrics_data:
            ax = axes[2]
            bars = ax.bar(model_names, metrics_data['max_drawdown'],
                         color=[self.colors['loss']] * n_models, alpha=0.7)
            ax.set_title('Maximum Drawdown Comparison', fontsize=14, fontweight='bold')
            ax.set_ylabel('Max Drawdown (%)')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)
            ax.invert_yaxis()  # 回撤向下显示
            
            for bar, value in zip(bars, metrics_data['max_drawdown']):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.2f}%', ha='center', va='top')
        
        # 4. 胜率对比
        if 'win_rate' in metrics_data:
            ax = axes[3]
            bars = ax.bar(model_names, [w*100 for w in metrics_data['win_rate']],
                         color=self.comparison_colors[:n_models], alpha=0.7)
            ax.set_title('Win Rate Comparison', fontsize=14, fontweight='bold')
            ax.set_ylabel('Win Rate (%)')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)
            ax.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='50% Baseline')
            ax.legend()
            
            for bar, value in zip(bars, [w*100 for w in metrics_data['win_rate']]):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.1f}%', ha='center', va='bottom')
        
        # 5. 雷达图 - 综合对比
        ax = axes[4]
        radar_metrics = ['mean_return', 'sharpe_ratio', 'win_rate', 'calmar_ratio']
        available_radar_metrics = [m for m in radar_metrics if m in metrics_data]
        
        if len(available_radar_metrics) >= 3:
            self._draw_radar_chart(ax, models_data, model_names, available_radar_metrics)
            ax.set_title('Comprehensive Performance Radar Chart', fontsize=14, fontweight='bold')
        else:
            ax.text(0.5, 0.5, '数据不足\n无法绘制雷达图', 
                   transform=ax.transAxes, ha='center', va='center')
        
        # 6. 波动率-收益率散点图
        ax = axes[5]
        if 'mean_return' in metrics_data and 'std_return' in metrics_data:
            for i, model_name in enumerate(model_names):
                ax.scatter(metrics_data['std_return'][i], metrics_data['mean_return'][i],
                          color=self.comparison_colors[i], s=100, alpha=0.7,
                          label=model_name)
                ax.annotate(model_name, 
                           (metrics_data['std_return'][i], metrics_data['mean_return'][i]),
                           xytext=(5, 5), textcoords='offset points', fontsize=9)
            
            ax.set_title('Risk-Return Distribution', fontsize=14, fontweight='bold')
            ax.set_xlabel('Volatility (%)')
            ax.set_ylabel('Average Return (%)')
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        # 7-9. 统计摘要表格
        for idx in [6, 7, 8]:
            axes[idx].axis('off')
        
        # 创建排名表
        self._create_ranking_table(axes[6], metrics_data, model_names)
        
        # 创建详细统计表
        self._create_detailed_stats_table(axes[7], models_data, model_names)
        
        # 创建风险指标表
        self._create_risk_metrics_table(axes[8], metrics_data, model_names)
        
        return self.save_figure(fig, 'performance_metrics_comparison', 'comparison')
    
    def compare_return_curves(self, 
                             models_returns: Dict[str, List[float]],
                             time_periods: Optional[List] = None,
                             cumulative: bool = True) -> List[str]:
        """
        对比多个模型的收益曲线
        
        Args:
            models_returns: 模型收益数据 {'Model_A': [returns...], ...}
            time_periods: 时间周期列表
            cumulative: 是否显示累积收益
            
        Returns:
            List[str]: 保存的文件路径
        """
        fig, axes = self.create_figure(figsize=(16, 10), subplots=(2, 1))
        
        model_names = list(models_returns.keys())
        
        # 创建时间轴
        max_length = max(len(returns) for returns in models_returns.values())
        if time_periods is None:
            time_periods = list(range(max_length))
        
        # 1. 收益曲线对比
        ax = axes[0]
        
        for i, (model_name, returns) in enumerate(models_returns.items()):
            if cumulative:
                # 累积收益
                cum_returns = np.cumsum(returns)
                ax.plot(time_periods[:len(cum_returns)], cum_returns,
                       color=self.comparison_colors[i % len(self.comparison_colors)],
                       linewidth=2, label=model_name, alpha=0.8)
                ylabel = '累积收益率 (%)'
                title = '累积收益曲线对比'
            else:
                # 周期收益
                ax.plot(time_periods[:len(returns)], returns,
                       color=self.comparison_colors[i % len(self.comparison_colors)],
                       linewidth=1.5, label=model_name, alpha=0.7)
                ylabel = '收益率 (%)'
                title = '周期收益曲线对比'
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Time Period')
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        # 2. 滚动夏普比率对比
        ax = axes[1]
        window = 50  # 50期滚动窗口
        
        for i, (model_name, returns) in enumerate(models_returns.items()):
            if len(returns) > window:
                rolling_sharpe = []
                for j in range(window, len(returns)):
                    window_returns = returns[j-window:j]
                    mean_ret = np.mean(window_returns)
                    std_ret = np.std(window_returns)
                    sharpe = mean_ret / std_ret if std_ret > 0 else 0
                    rolling_sharpe.append(sharpe)
                
                ax.plot(time_periods[window:window+len(rolling_sharpe)], rolling_sharpe,
                       color=self.comparison_colors[i % len(self.comparison_colors)],
                       linewidth=2, label=model_name, alpha=0.8)
        
        ax.set_title(f'Rolling Sharpe Ratio Comparison ({window}-Period Window)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Time Period')
        ax.set_ylabel('Rolling Sharpe Ratio')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='Excellence Line (1.0)')
        
        return self.save_figure(fig, 'return_curves_comparison', 'comparison')
    
    def compare_risk_metrics(self, 
                            models_data: Dict[str, Dict[str, Any]]) -> List[str]:
        """
        对比多个模型的风险指标
        
        Args:
            models_data: 模型数据字典
            
        Returns:
            List[str]: 保存的文件路径
        """
        fig, axes = self.create_figure(figsize=(16, 12), subplots=(2, 2))
        
        model_names = list(models_data.keys())
        n_models = len(model_names)
        
        # 计算各种风险指标
        risk_metrics = {}
        
        for model_name, data in models_data.items():
            returns = data.get('returns', [])
            if not returns:
                continue
            
            risk_metrics[model_name] = {
                'volatility': np.std(returns),
                'downside_deviation': np.std([r for r in returns if r < 0]),
                'var_95': np.percentile(returns, 5),
                'cvar_95': np.mean([r for r in returns if r <= np.percentile(returns, 5)]),
                'max_drawdown': self._calculate_max_drawdown(data.get('portfolio_values', [])),
                'calmar_ratio': np.mean(returns) / abs(self._calculate_max_drawdown(data.get('portfolio_values', []))) if self._calculate_max_drawdown(data.get('portfolio_values', [])) != 0 else 0
            }
        
        # 1. 风险指标雷达图
        ax = axes[0]
        risk_metrics_names = ['volatility', 'downside_deviation', 'max_drawdown']
        self._draw_risk_radar_chart(ax, risk_metrics, model_names, risk_metrics_names)
        ax.set_title('Risk Indicators Radar Chart', fontsize=14, fontweight='bold')
        
        # 2. VaR和CVaR对比
        ax = axes[1]
        var_values = [risk_metrics[name]['var_95'] for name in model_names]
        cvar_values = [risk_metrics[name]['cvar_95'] for name in model_names]
        
        x = np.arange(n_models)
        width = 0.35
        
        bars1 = ax.bar(x - width/2, var_values, width, label='VaR (95%)', 
                      color=self.colors['warning'], alpha=0.7)
        bars2 = ax.bar(x + width/2, cvar_values, width, label='CVaR (95%)', 
                      color=self.colors['loss'], alpha=0.7)
        
        ax.set_title('VaR and CVaR Comparison (95% Confidence)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Model')
        ax.set_ylabel('Loss Value (%)')
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 添加数值标签
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}', ha='center', va='bottom')
        
        # 3. 下行风险对比
        ax = axes[2]
        downside_devs = [risk_metrics[name]['downside_deviation'] for name in model_names]
        
        bars = ax.bar(model_names, downside_devs, 
                     color=[self.colors['loss']] * n_models, alpha=0.7)
        ax.set_title('Downside Standard Deviation Comparison', fontsize=14, fontweight='bold')
        ax.set_ylabel('Downside Std Dev (%)')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
        
        for bar, value in zip(bars, downside_devs):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.3f}', ha='center', va='bottom')
        
        # 4. Calmar比率对比
        ax = axes[3]
        calmar_ratios = [risk_metrics[name]['calmar_ratio'] for name in model_names]
        
        bars = ax.bar(model_names, calmar_ratios,
                     color=self.comparison_colors[:n_models], alpha=0.7)
        ax.set_title('Calmar Ratio Comparison', fontsize=14, fontweight='bold')
        ax.set_ylabel('Calmar Ratio')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Excellence Line (1.0)')
        ax.legend()
        
        for bar, value in zip(bars, calmar_ratios):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.3f}', ha='center', va='bottom')
        
        return self.save_figure(fig, 'risk_metrics_comparison', 'comparison')
    
    def statistical_significance_test(self, 
                                     models_returns: Dict[str, List[float]],
                                     test_type: str = 'welch') -> List[str]:
        """
        进行统计显著性检验
        
        Args:
            models_returns: 模型收益数据
            test_type: 检验类型 ('welch', 'ttest', 'wilcoxon')
            
        Returns:
            List[str]: 保存的文件路径
        """
        fig, axes = self.create_figure(figsize=(14, 10), subplots=(2, 2))
        
        model_names = list(models_returns.keys())
        n_models = len(model_names)
        
        # 1. 收益分布对比
        ax = axes[0]
        
        for i, (model_name, returns) in enumerate(models_returns.items()):
            ax.hist(returns, bins=30, alpha=0.6, 
                   color=self.comparison_colors[i % len(self.comparison_colors)],
                   label=model_name, density=True)
        
        ax.set_title('Return Distribution Comparison', fontsize=14, fontweight='bold')
        ax.set_xlabel('Return (%)')
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. 箱线图对比
        ax = axes[1]
        
        returns_data = [models_returns[name] for name in model_names]
        box_plot = ax.boxplot(returns_data, labels=model_names, patch_artist=True)
        
        # 设置箱线图颜色
        for patch, color in zip(box_plot['boxes'], self.comparison_colors[:n_models]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_title('Return Distribution Box Plot', fontsize=14, fontweight='bold')
        ax.set_ylabel('Return (%)')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
        
        # 3. 配对t检验热力图
        ax = axes[2]
        
        p_values = np.ones((n_models, n_models))
        for i in range(n_models):
            for j in range(n_models):
                if i != j:
                    if test_type == 'welch':
                        _, p_val = stats.ttest_ind(models_returns[model_names[i]], 
                                                 models_returns[model_names[j]], 
                                                 equal_var=False)
                    elif test_type == 'ttest':
                        _, p_val = stats.ttest_ind(models_returns[model_names[i]], 
                                                 models_returns[model_names[j]])
                    elif test_type == 'wilcoxon':
                        _, p_val = stats.mannwhitneyu(models_returns[model_names[i]], 
                                                    models_returns[model_names[j]], 
                                                    alternative='two-sided')
                    p_values[i, j] = p_val
        
        im = ax.imshow(p_values, cmap='RdYlGn', vmin=0, vmax=0.1)
        ax.set_xticks(range(n_models))
        ax.set_yticks(range(n_models))
        ax.set_xticklabels(model_names, rotation=45)
        ax.set_yticklabels(model_names)
        ax.set_title(f'统计显著性检验 P值热力图\n({test_type.upper()}检验)', fontsize=14, fontweight='bold')
        
        # 添加文本注释
        for i in range(n_models):
            for j in range(n_models):
                text = f'{p_values[i, j]:.3f}' if i != j else '-'
                ax.text(j, i, text, ha='center', va='center')
        
        plt.colorbar(im, ax=ax, label='P值')
        
        # 4. 统计摘要表
        ax = axes[3]
        ax.axis('off')
        
        # 计算统计指标
        stats_data = []
        for model_name in model_names:
            returns = models_returns[model_name]
            stats_data.append([
                model_name,
                f'{np.mean(returns):.3f}%',
                f'{np.std(returns):.3f}%',
                f'{stats.skew(returns):.3f}',
                f'{stats.kurtosis(returns):.3f}',
                f'{stats.jarque_bera(returns)[1]:.3f}'  # JB test p-value
            ])
        
        table = ax.table(cellText=stats_data,
                        colLabels=['模型', '均值', '标准差', '偏度', '峰度', 'JB检验P值'],
                        cellLoc='center',
                        loc='center',
                        colWidths=[0.2, 0.15, 0.15, 0.15, 0.15, 0.2])
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 2)
        
        ax.set_title('统计特征摘要', fontsize=14, fontweight='bold', pad=20)
        
        return self.save_figure(fig, 'statistical_significance_test', 'comparison')
    
    def _calculate_metric(self, model_data: Dict[str, Any], metric: str) -> float:
        """计算指定的性能指标"""
        returns = model_data.get('returns', [])
        portfolio_values = model_data.get('portfolio_values', [])
        
        if not returns:
            return 0.0
        
        if metric == 'mean_return':
            return np.mean(returns)
        elif metric == 'std_return':
            return np.std(returns)
        elif metric == 'sharpe_ratio':
            return np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
        elif metric == 'max_drawdown':
            return self._calculate_max_drawdown(portfolio_values)
        elif metric == 'win_rate':
            return len([r for r in returns if r > 0]) / len(returns)
        elif metric == 'calmar_ratio':
            max_dd = self._calculate_max_drawdown(portfolio_values)
            return abs(np.mean(returns) / max_dd) if max_dd != 0 else float('inf')
        elif metric == 'sortino_ratio':
            downside_returns = [r for r in returns if r < 0]
            downside_deviation = np.std(downside_returns) if downside_returns else 0
            return np.mean(returns) / downside_deviation if downside_deviation > 0 else 0
        else:
            return 0.0
    
    def _calculate_max_drawdown(self, portfolio_values: List[float]) -> float:
        """计算最大回撤"""
        if not portfolio_values:
            return 0.0
        
        peak = portfolio_values[0]
        max_drawdown = 0.0
        
        for value in portfolio_values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            max_drawdown = max(max_drawdown, drawdown)
        
        return max_drawdown * 100  # 转换为百分比
    
    def _draw_radar_chart(self, ax, models_data, model_names, metrics):
        """绘制雷达图"""
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # 闭合图形
        
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        
        # 标准化数据
        normalized_data = {}
        for metric in metrics:
            values = []
            for model_name in model_names:
                value = self._calculate_metric(models_data[model_name], metric)
                values.append(value)
            
            # 最小-最大标准化
            if max(values) != min(values):
                normalized_values = [(v - min(values)) / (max(values) - min(values)) 
                                   for v in values]
            else:
                normalized_values = [0.5] * len(values)
            
            for i, model_name in enumerate(model_names):
                if model_name not in normalized_data:
                    normalized_data[model_name] = []
                normalized_data[model_name].append(normalized_values[i])
        
        # 绘制每个模型
        for i, model_name in enumerate(model_names):
            values = normalized_data[model_name] + [normalized_data[model_name][0]]
            ax.plot(angles, values, 'o-', linewidth=2, 
                   color=self.comparison_colors[i % len(self.comparison_colors)],
                   label=model_name)
            ax.fill(angles, values, alpha=0.25, 
                   color=self.comparison_colors[i % len(self.comparison_colors)])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics])
        ax.set_ylim(0, 1)
        ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
        ax.grid(True)
    
    def _draw_risk_radar_chart(self, ax, risk_metrics, model_names, metrics_names):
        """绘制风险指标雷达图"""
        angles = np.linspace(0, 2 * np.pi, len(metrics_names), endpoint=False).tolist()
        angles += angles[:1]
        
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        
        # 风险指标需要反向标准化（越小越好）
        normalized_data = {}
        for metric in metrics_names:
            values = [risk_metrics[name][metric] for name in model_names]
            
            if max(values) != min(values):
                # 反向标准化：值越小，标准化后越大
                normalized_values = [(max(values) - v) / (max(values) - min(values)) 
                                   for v in values]
            else:
                normalized_values = [0.5] * len(values)
            
            for i, model_name in enumerate(model_names):
                if model_name not in normalized_data:
                    normalized_data[model_name] = []
                normalized_data[model_name].append(normalized_values[i])
        
        # 绘制每个模型
        for i, model_name in enumerate(model_names):
            values = normalized_data[model_name] + [normalized_data[model_name][0]]
            ax.plot(angles, values, 'o-', linewidth=2,
                   color=self.comparison_colors[i % len(self.comparison_colors)],
                   label=model_name)
            ax.fill(angles, values, alpha=0.25,
                   color=self.comparison_colors[i % len(self.comparison_colors)])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics_names])
        ax.set_ylim(0, 1)
        ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
        ax.grid(True)
    
    def _create_ranking_table(self, ax, metrics_data, model_names):
        """创建排名表"""
        ax.axis('off')
        ax.set_title('性能排名', fontsize=14, fontweight='bold')
        
        # 计算综合得分（简化版）
        scores = {}
        for i, model_name in enumerate(model_names):
            score = 0
            # 收益率权重40%
            if 'mean_return' in metrics_data:
                score += metrics_data['mean_return'][i] * 0.4
            # 夏普比率权重30%
            if 'sharpe_ratio' in metrics_data:
                score += metrics_data['sharpe_ratio'][i] * 30
            # 胜率权重30%
            if 'win_rate' in metrics_data:
                score += metrics_data['win_rate'][i] * 30
            
            scores[model_name] = score
        
        # 按得分排序
        sorted_models = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        # 创建表格数据
        table_data = []
        for rank, (model_name, score) in enumerate(sorted_models, 1):
            table_data.append([f'#{rank}', model_name, f'{score:.2f}'])
        
        table = ax.table(cellText=table_data,
                        colLabels=['排名', '模型', '综合得分'],
                        cellLoc='center',
                        loc='center')
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.8)
    
    def _create_detailed_stats_table(self, ax, models_data, model_names):
        """创建详细统计表"""
        ax.axis('off')
        ax.set_title('详细统计', fontsize=14, fontweight='bold')
        
        table_data = []
        for model_name in model_names:
            data = models_data[model_name]
            returns = data.get('returns', [])
            
            if returns:
                table_data.append([
                    model_name,
                    f'{np.mean(returns):.2f}%',
                    f'{np.std(returns):.2f}%',
                    f'{max(returns):.2f}%',
                    f'{min(returns):.2f}%'
                ])
        
        table = ax.table(cellText=table_data,
                        colLabels=['模型', '均值', '标准差', '最大', '最小'],
                        cellLoc='center',
                        loc='center')
        
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
    
    def _create_risk_metrics_table(self, ax, metrics_data, model_names):
        """创建风险指标表"""
        ax.axis('off')
        ax.set_title('风险指标', fontsize=14, fontweight='bold')
        
        table_data = []
        for i, model_name in enumerate(model_names):
            row = [model_name]
            
            if 'max_drawdown' in metrics_data:
                row.append(f'{metrics_data["max_drawdown"][i]:.2f}%')
            else:
                row.append('N/A')
            
            if 'sharpe_ratio' in metrics_data:
                row.append(f'{metrics_data["sharpe_ratio"][i]:.3f}')
            else:
                row.append('N/A')
            
            if 'calmar_ratio' in metrics_data:
                row.append(f'{metrics_data["calmar_ratio"][i]:.3f}')
            else:
                row.append('N/A')
            
            table_data.append(row)
        
        table = ax.table(cellText=table_data,
                        colLabels=['模型', '最大回撤', '夏普比率', 'Calmar比率'],
                        cellLoc='center',
                        loc='center')
        
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)