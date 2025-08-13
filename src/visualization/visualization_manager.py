"""
统一可视化管理器

协调和管理所有可视化组件，提供统一的接口和高级可视化功能。
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
import logging
from pathlib import Path

from .base_visualizer import BaseVisualizer
from .training_visualizer import TrainingVisualizer
from .evaluation_visualizer import EvaluationVisualizer
from .forex_visualizer import ForexVisualizer
from .comparison_visualizer import ComparisonVisualizer


class VisualizationManager:
    """
    统一可视化管理器
    
    负责协调所有可视化组件，提供高级可视化功能，包括：
    - 训练过程可视化
    - 评估结果分析
    - Forex专用图表
    - 多模型对比
    - 报告生成
    - 批量图表生成
    """
    
    def __init__(self, 
                 output_dir: str = "visualizations",
                 config: Optional[Dict[str, Any]] = None):
        """
        初始化可视化管理器
        
        Args:
            output_dir: 输出目录
            config: 可视化配置
        """
        self.output_dir = Path(output_dir)
        self.config = config or {}
        
        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置日志
        self.logger = logging.getLogger('VisualizationManager')
        
        # 初始化各个可视化组件
        viz_config = {
            'output_dir': output_dir,
            'style_theme': self.config.get('style_theme', 'seaborn-v0_8'),
            'figure_size': tuple(self.config.get('figure_size', [12, 8])),
            'dpi': self.config.get('dpi', 300),
            'save_formats': self.config.get('save_formats', ['png', 'pdf'])
        }
        
        self.training_viz = TrainingVisualizer(**viz_config)
        self.evaluation_viz = EvaluationVisualizer(**viz_config)
        self.forex_viz = ForexVisualizer(**viz_config)
        self.comparison_viz = ComparisonVisualizer(**viz_config)
        
        # 会话信息
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.generated_charts = []
        
        self.logger.info(f"可视化管理器初始化完成 - 会话ID: {self.session_id}")
    
    def generate_training_visualizations(self,
                                        training_data: Dict[str, Any],
                                        experiment_name: str = "",
                                        detailed: bool = True) -> Dict[str, List[str]]:
        """
        生成训练过程可视化
        
        Args:
            training_data: 训练数据
                格式: {
                    'episode_rewards': [...],
                    'portfolio_values': [...],
                    'actions_history': [...],
                    'training_metrics': {...}  # 可选
                }
            experiment_name: 实验名称
            detailed: 是否生成详细图表
            
        Returns:
            Dict[str, List[str]]: 生成的图表文件路径
        """
        self.logger.info(f"开始生成训练可视化 - 实验: {experiment_name}")
        
        generated_files = {}
        
        try:
            # 1. 奖励曲线分析
            if 'episode_rewards' in training_data:
                files = self.training_viz.plot_reward_curves(
                    episode_rewards=training_data['episode_rewards'],
                    smoothing_window=self.config.get('reward_smoothing_window', 100)
                )
                generated_files['reward_curves'] = files
                self.generated_charts.extend(files)
                
            # 2. 投资组合演化
            if 'portfolio_values' in training_data:
                initial_balance = training_data.get('initial_balance', 10000.0)
                files = self.training_viz.plot_portfolio_evolution(
                    portfolio_values=training_data['portfolio_values'],
                    initial_balance=initial_balance
                )
                generated_files['portfolio_evolution'] = files
                self.generated_charts.extend(files)
                
            # 3. 交易动作分析
            if 'actions_history' in training_data and 'portfolio_values' in training_data:
                files = self.training_viz.plot_action_analysis(
                    actions_history=training_data['actions_history'],
                    portfolio_values=training_data['portfolio_values']
                )
                generated_files['action_analysis'] = files
                self.generated_charts.extend(files)
                
            # 4. 学习进度监控
            if 'training_metrics' in training_data and training_data['training_metrics']:
                files = self.training_viz.plot_learning_progress(
                    training_metrics=training_data['training_metrics']
                )
                generated_files['learning_progress'] = files
                self.generated_charts.extend(files)
                
            # 5. 综合仪表板（如果需要详细分析）
            if detailed and all(key in training_data for key in ['episode_rewards', 'portfolio_values', 'actions_history']):
                files = self.training_viz.create_training_dashboard(
                    episode_rewards=training_data['episode_rewards'],
                    portfolio_values=training_data['portfolio_values'],
                    actions_history=training_data['actions_history'],
                    training_metrics=training_data.get('training_metrics'),
                    initial_balance=training_data.get('initial_balance', 10000.0)
                )
                generated_files['dashboard'] = files
                self.generated_charts.extend(files)
                
        except Exception as e:
            self.logger.error(f"生成训练可视化时出错: {e}")
            
        self.logger.info(f"训练可视化完成 - 生成了 {len(generated_files)} 类图表")
        return generated_files
    
    def generate_evaluation_visualizations(self,
                                          evaluation_data: Dict[str, Any],
                                          model_info: Dict[str, Any] = None,
                                          detailed: bool = True) -> Dict[str, List[str]]:
        """
        生成评估结果可视化
        
        Args:
            evaluation_data: 评估数据
                格式: {
                    'episode_data': [...],  # 每个episode的统计
                    'prices': [...],        # 价格序列
                    'actions': [...],       # 动作序列
                    'portfolio_values': [...] # 投资组合价值序列
                }
            model_info: 模型信息
            detailed: 是否生成详细图表
            
        Returns:
            Dict[str, List[str]]: 生成的图表文件路径
        """
        self.logger.info("开始生成评估可视化")
        
        generated_files = {}
        model_info = model_info or {}
        
        try:
            # 1. Episode性能分析
            if 'episode_data' in evaluation_data:
                files = self.evaluation_viz.plot_episode_performance(
                    episode_data=evaluation_data['episode_data'],
                    show_individual=detailed
                )
                generated_files['episode_performance'] = files
                self.generated_charts.extend(files)
                
            # 2. 交易信号分析
            if all(key in evaluation_data for key in ['prices', 'actions', 'portfolio_values']):
                files = self.evaluation_viz.plot_trading_signals(
                    prices=evaluation_data['prices'],
                    actions=evaluation_data['actions'],
                    portfolio_values=evaluation_data['portfolio_values'],
                    timestamps=evaluation_data.get('timestamps')
                )
                generated_files['trading_signals'] = files
                self.generated_charts.extend(files)
                
            # 3. 风险收益分析
            if 'episode_data' in evaluation_data:
                files = self.evaluation_viz.plot_risk_return_analysis(
                    episode_data=evaluation_data['episode_data'],
                    benchmark_return=evaluation_data.get('benchmark_return', 0.0)
                )
                generated_files['risk_return_analysis'] = files
                self.generated_charts.extend(files)
                
            # 4. 完整评估报告
            if detailed and 'episode_data' in evaluation_data:
                files = self.evaluation_viz.create_evaluation_report(
                    episode_data=evaluation_data['episode_data'],
                    model_info=model_info,
                    test_period=evaluation_data.get('test_period', "")
                )
                generated_files['evaluation_report'] = files
                self.generated_charts.extend(files)
                
        except Exception as e:
            self.logger.error(f"生成评估可视化时出错: {e}")
            
        self.logger.info(f"评估可视化完成 - 生成了 {len(generated_files)} 类图表")
        return generated_files
    
    def generate_forex_visualizations(self,
                                     forex_data: Dict[str, Any],
                                     currency_pair: str = "EURUSD",
                                     detailed: bool = True) -> Dict[str, List[str]]:
        """
        生成Forex专用可视化
        
        Args:
            forex_data: Forex数据
                格式: {
                    'ohlc_data': DataFrame,  # OHLC数据
                    'prices': [...],         # 价格序列
                    'actions': [...],        # 交易动作
                    'pip_size': 0.0001      # 点大小
                }
            currency_pair: 货币对名称
            detailed: 是否生成详细图表
            
        Returns:
            Dict[str, List[str]]: 生成的图表文件路径
        """
        self.logger.info(f"开始生成Forex可视化 - 货币对: {currency_pair}")
        
        generated_files = {}
        
        try:
            # 1. 蜡烛图（如果有OHLC数据）
            if 'ohlc_data' in forex_data:
                files = self.forex_viz.plot_candlestick_chart(
                    ohlc_data=forex_data['ohlc_data'],
                    actions=forex_data.get('actions'),
                    title=f'{currency_pair} 蜡烛图分析',
                    show_volume='Volume' in forex_data['ohlc_data'].columns
                )
                generated_files['candlestick_chart'] = files
                self.generated_charts.extend(files)
                
            # 2. 点数分析
            if 'prices' in forex_data and 'actions' in forex_data:
                files = self.forex_viz.plot_pip_analysis(
                    prices=forex_data['prices'],
                    actions=forex_data['actions'],
                    pip_size=forex_data.get('pip_size', 0.0001),
                    currency_pair=currency_pair
                )
                generated_files['pip_analysis'] = files
                self.generated_charts.extend(files)
                
            # 3. 趋势分析
            if 'prices' in forex_data:
                files = self.forex_viz.plot_trend_analysis(
                    prices=forex_data['prices'],
                    trend_periods=forex_data.get('trend_periods', [10, 20, 50]),
                    currency_pair=currency_pair
                )
                generated_files['trend_analysis'] = files
                self.generated_charts.extend(files)
                
            # 4. 支撑阻力分析（如果有OHLC数据且需要详细分析）
            if detailed and 'ohlc_data' in forex_data:
                files = self.forex_viz.plot_support_resistance(
                    ohlc_data=forex_data['ohlc_data'],
                    lookback_period=forex_data.get('sr_lookback_period', 20)
                )
                generated_files['support_resistance'] = files
                self.generated_charts.extend(files)
                
            # 5. 交易时段分析（如果有时间索引）
            if detailed and 'ohlc_data' in forex_data:
                if isinstance(forex_data['ohlc_data'].index, pd.DatetimeIndex):
                    files = self.forex_viz.plot_trading_session_analysis(
                        ohlc_data=forex_data['ohlc_data'],
                        timezone=forex_data.get('timezone', 'UTC')
                    )
                    generated_files['trading_session_analysis'] = files
                    self.generated_charts.extend(files)
                    
        except Exception as e:
            self.logger.error(f"生成Forex可视化时出错: {e}")
            
        self.logger.info(f"Forex可视化完成 - 生成了 {len(generated_files)} 类图表")
        return generated_files
    
    def generate_comparison_visualizations(self,
                                          models_data: Dict[str, Dict[str, Any]],
                                          comparison_type: str = 'performance',
                                          detailed: bool = True) -> Dict[str, List[str]]:
        """
        生成模型对比可视化
        
        Args:
            models_data: 多个模型的数据
                格式: {'Model_A': {...}, 'Model_B': {...}}
            comparison_type: 对比类型 ('performance', 'returns', 'risk', 'statistical')
            detailed: 是否生成详细图表
            
        Returns:
            Dict[str, List[str]]: 生成的图表文件路径
        """
        self.logger.info(f"开始生成对比可视化 - 类型: {comparison_type}")
        
        generated_files = {}
        
        try:
            if comparison_type in ['performance', 'all']:
                # 性能指标对比
                files = self.comparison_viz.compare_performance_metrics(
                    models_data=models_data
                )
                generated_files['performance_metrics'] = files
                self.generated_charts.extend(files)
                
            if comparison_type in ['returns', 'all']:
                # 收益曲线对比
                models_returns = {name: data.get('returns', []) 
                                for name, data in models_data.items()}
                if any(returns for returns in models_returns.values()):
                    files = self.comparison_viz.compare_return_curves(
                        models_returns=models_returns,
                        cumulative=True
                    )
                    generated_files['return_curves'] = files
                    self.generated_charts.extend(files)
                    
            if comparison_type in ['risk', 'all']:
                # 风险指标对比
                files = self.comparison_viz.compare_risk_metrics(
                    models_data=models_data
                )
                generated_files['risk_metrics'] = files
                self.generated_charts.extend(files)
                
            if detailed and comparison_type in ['statistical', 'all']:
                # 统计显著性检验
                models_returns = {name: data.get('returns', []) 
                                for name, data in models_data.items()}
                if any(returns for returns in models_returns.values()):
                    files = self.comparison_viz.statistical_significance_test(
                        models_returns=models_returns
                    )
                    generated_files['statistical_test'] = files
                    self.generated_charts.extend(files)
                    
        except Exception as e:
            self.logger.error(f"生成对比可视化时出错: {e}")
            
        self.logger.info(f"对比可视化完成 - 生成了 {len(generated_files)} 类图表")
        return generated_files
    
    def create_comprehensive_report(self,
                                   report_data: Dict[str, Any],
                                   report_name: str = "TensorTrade_Report",
                                   include_html: bool = True) -> Dict[str, str]:
        """
        创建综合报告
        
        Args:
            report_data: 报告数据
            report_name: 报告名称
            include_html: 是否生成HTML报告
            
        Returns:
            Dict[str, str]: 报告文件路径
        """
        self.logger.info(f"开始创建综合报告: {report_name}")
        
        generated_reports = {}
        
        try:
            # 创建报告目录
            report_dir = self.output_dir / f"reports/{report_name}_{self.session_id}"
            report_dir.mkdir(parents=True, exist_ok=True)
            
            # 生成所有可视化
            all_visualizations = {}
            
            # 训练可视化
            if 'training_data' in report_data:
                training_viz = self.generate_training_visualizations(
                    training_data=report_data['training_data'],
                    experiment_name=report_data.get('experiment_name', ''),
                    detailed=True
                )
                all_visualizations.update(training_viz)
                
            # 评估可视化
            if 'evaluation_data' in report_data:
                evaluation_viz = self.generate_evaluation_visualizations(
                    evaluation_data=report_data['evaluation_data'],
                    model_info=report_data.get('model_info', {}),
                    detailed=True
                )
                all_visualizations.update(evaluation_viz)
                
            # Forex可视化
            if 'forex_data' in report_data:
                forex_viz = self.generate_forex_visualizations(
                    forex_data=report_data['forex_data'],
                    currency_pair=report_data.get('currency_pair', 'EURUSD'),
                    detailed=True
                )
                all_visualizations.update(forex_viz)
                
            # 对比可视化
            if 'comparison_data' in report_data:
                comparison_viz = self.generate_comparison_visualizations(
                    models_data=report_data['comparison_data'],
                    comparison_type='all',
                    detailed=True
                )
                all_visualizations.update(comparison_viz)
            
            # 生成Markdown报告
            markdown_file = self._generate_markdown_report(
                report_dir, report_name, report_data, all_visualizations
            )
            generated_reports['markdown'] = str(markdown_file)
            
            # 生成HTML报告（如果需要）
            if include_html:
                html_file = self._generate_html_report(
                    report_dir, report_name, report_data, all_visualizations
                )
                generated_reports['html'] = str(html_file)
                
        except Exception as e:
            self.logger.error(f"创建综合报告时出错: {e}")
            
        self.logger.info(f"综合报告创建完成: {generated_reports}")
        return generated_reports
    
    def get_session_summary(self) -> Dict[str, Any]:
        """
        获取会话摘要
        
        Returns:
            Dict[str, Any]: 会话摘要信息
        """
        return {
            'session_id': self.session_id,
            'total_charts_generated': len(self.generated_charts),
            'chart_files': self.generated_charts,
            'output_directory': str(self.output_dir),
            'config': self.config
        }
    
    def _generate_markdown_report(self, report_dir, report_name, report_data, visualizations):
        """生成Markdown报告"""
        markdown_file = report_dir / f"{report_name}.md"
        
        with open(markdown_file, 'w', encoding='utf-8') as f:
            f.write(f"# {report_name}\n\n")
            f.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**会话ID**: {self.session_id}\n\n")
            
            # 实验信息
            if 'experiment_name' in report_data:
                f.write(f"**实验名称**: {report_data['experiment_name']}\n\n")
            
            if 'model_info' in report_data:
                f.write("## 模型信息\n\n")
                model_info = report_data['model_info']
                for key, value in model_info.items():
                    f.write(f"- **{key}**: {value}\n")
                f.write("\n")
            
            # 图表目录
            f.write("## 图表目录\n\n")
            for category, files in visualizations.items():
                if files:
                    f.write(f"### {category.replace('_', ' ').title()}\n\n")
                    for file_path in files:
                        file_name = Path(file_path).name
                        f.write(f"![{file_name}]({file_path})\n\n")
            
            # 统计摘要
            f.write("## 统计摘要\n\n")
            f.write(f"- 生成图表总数: {len(self.generated_charts)}\n")
            f.write(f"- 图表类别数: {len(visualizations)}\n")
            f.write(f"- 输出目录: {self.output_dir}\n")
        
        return markdown_file
    
    def _generate_html_report(self, report_dir, report_name, report_data, visualizations):
        """生成HTML报告"""
        html_file = report_dir / f"{report_name}.html"
        
        html_content = f"""
        <!DOCTYPE html>
        <html lang="zh-CN">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{report_name}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
                .header {{ background-color: #f4f4f4; padding: 20px; border-radius: 5px; }}
                .chart-section {{ margin: 30px 0; }}
                .chart-category {{ border-left: 3px solid #007cba; padding-left: 20px; margin: 20px 0; }}
                img {{ max-width: 100%; height: auto; border: 1px solid #ddd; margin: 10px 0; }}
                .summary {{ background-color: #e7f3ff; padding: 15px; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>{report_name}</h1>
                <p><strong>生成时间:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p><strong>会话ID:</strong> {self.session_id}</p>
        """
        
        if 'experiment_name' in report_data:
            html_content += f"<p><strong>实验名称:</strong> {report_data['experiment_name']}</p>"
        
        html_content += "</div>\n"
        
        # 模型信息
        if 'model_info' in report_data:
            html_content += "<h2>模型信息</h2>\n<ul>\n"
            for key, value in report_data['model_info'].items():
                html_content += f"<li><strong>{key}:</strong> {value}</li>\n"
            html_content += "</ul>\n"
        
        # 图表展示
        html_content += "<h2>可视化图表</h2>\n"
        for category, files in visualizations.items():
            if files:
                html_content += f'<div class="chart-category">\n'
                html_content += f"<h3>{category.replace('_', ' ').title()}</h3>\n"
                for file_path in files:
                    file_name = Path(file_path).name
                    html_content += f'<img src="{file_path}" alt="{file_name}" title="{file_name}">\n'
                html_content += "</div>\n"
        
        # 统计摘要
        html_content += f"""
            <div class="summary">
                <h2>统计摘要</h2>
                <ul>
                    <li>生成图表总数: {len(self.generated_charts)}</li>
                    <li>图表类别数: {len(visualizations)}</li>
                    <li>输出目录: {self.output_dir}</li>
                </ul>
            </div>
        </body>
        </html>
        """
        
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return html_file
    
    def cleanup_old_files(self, days_old: int = 7):
        """清理旧的可视化文件"""
        cutoff_time = datetime.now() - timedelta(days=days_old)
        
        removed_count = 0
        for file_path in self.output_dir.rglob('*'):
            if file_path.is_file():
                if datetime.fromtimestamp(file_path.stat().st_mtime) < cutoff_time:
                    try:
                        file_path.unlink()
                        removed_count += 1
                    except Exception as e:
                        self.logger.warning(f"无法删除文件 {file_path}: {e}")
        
        self.logger.info(f"清理完成 - 删除了 {removed_count} 个旧文件")
        
    def __enter__(self):
        """上下文管理器入口"""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        # 清理资源
        if hasattr(self, 'training_viz'):
            self.training_viz.close_all_figures()
        if hasattr(self, 'evaluation_viz'):
            self.evaluation_viz.close_all_figures()
        if hasattr(self, 'forex_viz'):
            self.forex_viz.close_all_figures()
        if hasattr(self, 'comparison_viz'):
            self.comparison_viz.close_all_figures()