"""
Walk Forward 验证模块
基于时间序列的前向验证，用于评估交易模型的稳健性

主要功能:
1. 时间序列数据分割和滚动窗口验证
2. 避免数据泄露的前向验证流程
3. 交易性能指标计算和分析
4. 验证结果汇总和可视化准备
5. 模型稳定性评估
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
import warnings

from ..utils.logger import setup_logger, get_default_log_file
from ..utils.config import Config


class WalkForwardValidator:
    """
    Walk Forward 验证器
    
    实现基于时间序列的前向验证方法，确保模型评估的可靠性:
    - 严格按时间顺序分割数据，避免未来信息泄露
    - 滚动窗口训练和测试
    - 多种性能指标计算
    - 稳定性分析
    """
    
    def __init__(
        self,
        config: Optional[Config] = None,
        train_ratio: float = 0.7,
        test_ratio: float = 0.2,
        validation_ratio: float = 0.1,
        step_ratio: float = 0.1,
        min_train_size: int = 100
    ):
        """
        初始化验证器
        
        Args:
            config: 配置对象
            train_ratio: 训练集比例
            test_ratio: 测试集比例  
            validation_ratio: 验证集比例
            step_ratio: 每次前进的步长比例
            min_train_size: 最小训练集大小
        """
        self.config = config or Config()
        
        # 验证比例参数
        total_ratio = train_ratio + test_ratio + validation_ratio
        if abs(total_ratio - 1.0) > 1e-6:
            raise ValueError(f"train_ratio + test_ratio + validation_ratio 必须等于1.0，当前为: {total_ratio}")
        
        self.train_ratio = train_ratio
        self.test_ratio = test_ratio
        self.validation_ratio = validation_ratio
        self.step_ratio = step_ratio
        self.min_train_size = min_train_size
        
        # 初始化日志
        self.logger = setup_logger(
            name="WalkForwardValidator",
            level="INFO",
            log_file=get_default_log_file("walk_forward_validator")
        )
        
        # 验证状态
        self.validation_results = []
        self.summary_stats = {}
        self.data_splits = []
        
        self.logger.info("WalkForwardValidator 初始化完成")
        self.logger.info(f"分割比例 - 训练: {train_ratio}, 测试: {test_ratio}, 验证: {validation_ratio}")
        self.logger.info(f"步长比例: {step_ratio}, 最小训练集: {min_train_size}")
    
    def calculate_split_sizes(self, total_size: int) -> Tuple[int, int, int, int]:
        """
        计算各数据集的大小
        
        Args:
            total_size: 总数据大小
            
        Returns:
            Tuple[int, int, int, int]: (训练集大小, 测试集大小, 验证集大小, 步长大小)
        """
        train_size = max(int(total_size * self.train_ratio), self.min_train_size)
        test_size = int(total_size * self.test_ratio)
        validation_size = int(total_size * self.validation_ratio)
        step_size = max(int(total_size * self.step_ratio), 1)
        
        # 确保总大小不超过数据集
        window_size = train_size + test_size + validation_size
        if window_size > total_size:
            # 按比例缩减
            scale = total_size / window_size
            train_size = max(int(train_size * scale), self.min_train_size)
            test_size = int(test_size * scale)
            validation_size = total_size - train_size - test_size
            
        return train_size, test_size, validation_size, step_size
    
    def create_time_splits(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        创建时间序列数据分割
        
        Args:
            data: 输入数据（必须有时间索引）
            
        Returns:
            List[Dict[str, Any]]: 数据分割列表
        """
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("数据必须有DatetimeIndex作为索引")
        
        total_size = len(data)
        train_size, test_size, validation_size, step_size = self.calculate_split_sizes(total_size)
        
        window_size = train_size + test_size + validation_size
        splits = []
        
        self.logger.info(f"数据总长度: {total_size}")
        self.logger.info(f"窗口大小: {window_size} (训练: {train_size}, 测试: {test_size}, 验证: {validation_size})")
        self.logger.info(f"步长: {step_size}")
        
        # 生成滚动窗口
        start_idx = 0
        fold = 0
        
        while start_idx + window_size <= total_size:
            # 计算各个分割的索引
            train_start = start_idx
            train_end = train_start + train_size
            test_start = train_end
            test_end = test_start + test_size
            validation_start = test_end
            validation_end = validation_start + validation_size
            
            # 获取对应的时间戳
            split_info = {
                'fold': fold,
                'train_start_idx': train_start,
                'train_end_idx': train_end,
                'test_start_idx': test_start,
                'test_end_idx': test_end,
                'validation_start_idx': validation_start,
                'validation_end_idx': validation_end,
                'train_start_date': data.index[train_start],
                'train_end_date': data.index[train_end - 1],
                'test_start_date': data.index[test_start],
                'test_end_date': data.index[test_end - 1],
                'validation_start_date': data.index[validation_start],
                'validation_end_date': data.index[validation_end - 1],
                'window_size': window_size,
                'step_size': step_size
            }
            
            splits.append(split_info)
            
            self.logger.debug(f"Fold {fold}: {split_info['train_start_date']} -> {split_info['validation_end_date']}")
            
            fold += 1
            start_idx += step_size
        
        self.data_splits = splits
        self.logger.info(f"生成了 {len(splits)} 个时间窗口用于验证")
        
        return splits
    
    def get_split_data(
        self, 
        data: pd.DataFrame, 
        split_info: Dict[str, Any]
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        根据分割信息获取训练、测试、验证数据
        
        Args:
            data: 原始数据
            split_info: 分割信息
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: (训练数据, 测试数据, 验证数据)
        """
        train_data = data.iloc[split_info['train_start_idx']:split_info['train_end_idx']].copy()
        test_data = data.iloc[split_info['test_start_idx']:split_info['test_end_idx']].copy()
        validation_data = data.iloc[split_info['validation_start_idx']:split_info['validation_end_idx']].copy()
        
        # 验证数据时间连续性
        assert train_data.index[-1] < test_data.index[0], "训练和测试数据时间重叠"
        assert test_data.index[-1] < validation_data.index[0], "测试和验证数据时间重叠"
        
        return train_data, test_data, validation_data
    
    def validate(
        self,
        data: pd.DataFrame,
        create_agent_func: Callable,
        num_iterations: int = 50,
        custom_config: Optional[Dict[str, Any]] = None,
        evaluation_episodes: int = 5
    ) -> Dict[str, Any]:
        """
        执行Walk Forward验证
        
        Args:
            data: 训练数据（包含特征）
            create_agent_func: 创建智能体的函数
            num_iterations: 每折训练迭代次数
            custom_config: 自定义配置
            evaluation_episodes: 评估回合数
            
        Returns:
            Dict[str, Any]: 验证结果
        """
        start_time = datetime.now()
        self.logger.info("开始Walk Forward验证...")
        
        try:
            # 1. 创建时间分割
            splits = self.create_time_splits(data)
            
            if len(splits) == 0:
                raise ValueError("无法创建有效的时间分割，请检查数据大小和参数配置")
            
            # 2. 逐折验证
            fold_results = []
            
            for i, split_info in enumerate(splits):
                fold_start_time = datetime.now()
                self.logger.info(f"开始第 {i+1}/{len(splits)} 折验证")
                
                try:
                    # 获取分割数据
                    train_data, test_data, validation_data = self.get_split_data(data, split_info)
                    
                    self.logger.info(f"训练集: {len(train_data)} 条，时间范围: {train_data.index[0]} - {train_data.index[-1]}")
                    self.logger.info(f"测试集: {len(test_data)} 条，时间范围: {test_data.index[0]} - {test_data.index[-1]}")
                    self.logger.info(f"验证集: {len(validation_data)} 条，时间范围: {validation_data.index[0]} - {validation_data.index[-1]}")
                    
                    # 创建智能体
                    agent = create_agent_func()
                    
                    # 训练智能体
                    self.logger.info("开始训练...")
                    training_result = agent.train(
                        data=train_data,
                        num_iterations=num_iterations,
                        custom_config=custom_config
                    )
                    
                    # 在测试集上评估
                    self.logger.info("在测试集上评估...")
                    test_result = agent.evaluate_model(
                        data=test_data,
                        num_episodes=evaluation_episodes
                    )
                    
                    # 在验证集上评估
                    self.logger.info("在验证集上评估...")
                    validation_result = agent.evaluate_model(
                        data=validation_data,
                        num_episodes=evaluation_episodes
                    )
                    
                    # 清理资源
                    agent.cleanup()
                    
                    # 记录折结果
                    fold_duration = (datetime.now() - fold_start_time).total_seconds()
                    
                    fold_result = {
                        'fold': i,
                        'split_info': split_info,
                        'training_result': training_result,
                        'test_result': test_result,
                        'validation_result': validation_result,
                        'fold_duration': fold_duration,
                        'train_data_shape': train_data.shape,
                        'test_data_shape': test_data.shape,
                        'validation_data_shape': validation_data.shape
                    }
                    
                    fold_results.append(fold_result)
                    
                    self.logger.info(f"第 {i+1} 折完成，用时: {fold_duration:.2f}秒")
                    self.logger.info(f"测试集平均奖励: {test_result['mean_reward']:.4f}")
                    self.logger.info(f"验证集平均奖励: {validation_result['mean_reward']:.4f}")
                    
                except Exception as e:
                    self.logger.error(f"第 {i+1} 折验证失败: {e}")
                    # 继续下一折，不中断整个验证过程
                    continue
            
            # 3. 汇总结果
            if not fold_results:
                raise RuntimeError("所有折验证都失败了")
            
            self.validation_results = fold_results
            summary_result = self.summarize_results()
            
            total_duration = (datetime.now() - start_time).total_seconds()
            
            final_result = {
                'validation_type': 'walk_forward',
                'num_folds': len(fold_results),
                'total_folds_attempted': len(splits),
                'success_rate': len(fold_results) / len(splits),
                'fold_results': fold_results,
                'summary_statistics': summary_result,
                'total_duration': total_duration,
                'validation_config': {
                    'train_ratio': self.train_ratio,
                    'test_ratio': self.test_ratio,
                    'validation_ratio': self.validation_ratio,
                    'step_ratio': self.step_ratio,
                    'min_train_size': self.min_train_size,
                    'num_iterations': num_iterations,
                    'evaluation_episodes': evaluation_episodes
                },
                'data_info': {
                    'total_size': len(data),
                    'time_range': f"{data.index[0]} - {data.index[-1]}",
                    'features': list(data.columns)
                }
            }
            
            self.logger.info("Walk Forward验证完成")
            self.logger.info(f"总用时: {total_duration:.2f}秒")
            self.logger.info(f"成功率: {final_result['success_rate']:.2%}")
            
            return final_result
            
        except Exception as e:
            self.logger.error(f"Walk Forward验证失败: {e}")
            raise
    
    def calculate_metrics(self, rewards: List[float], actions: Optional[List[float]] = None) -> Dict[str, float]:
        """
        计算性能指标
        
        Args:
            rewards: 奖励列表
            actions: 动作列表（可选）
            
        Returns:
            Dict[str, float]: 性能指标字典
        """
        if not rewards or len(rewards) == 0:
            return {}
        
        rewards_array = np.array(rewards)
        
        # 基础指标
        metrics = {
            'total_reward': float(np.sum(rewards_array)),
            'mean_reward': float(np.mean(rewards_array)),
            'std_reward': float(np.std(rewards_array)),
            'min_reward': float(np.min(rewards_array)),
            'max_reward': float(np.max(rewards_array)),
            'median_reward': float(np.median(rewards_array)),
        }
        
        # 风险指标
        if len(rewards_array) > 1:
            # 计算累计收益
            cumulative_rewards = np.cumsum(rewards_array)
            
            # 最大回撤
            peak = np.maximum.accumulate(cumulative_rewards)
            drawdown = peak - cumulative_rewards
            max_drawdown = float(np.max(drawdown))
            
            # 夏普比率（假设无风险利率为0）
            if metrics['std_reward'] > 0:
                sharpe_ratio = metrics['mean_reward'] / metrics['std_reward']
            else:
                sharpe_ratio = 0.0
            
            # 胜率
            positive_rewards = rewards_array > 0
            win_rate = float(np.mean(positive_rewards)) if len(positive_rewards) > 0 else 0.0
            
            # 盈亏比
            winning_rewards = rewards_array[rewards_array > 0]
            losing_rewards = rewards_array[rewards_array < 0]
            
            avg_win = float(np.mean(winning_rewards)) if len(winning_rewards) > 0 else 0.0
            avg_loss = float(np.mean(np.abs(losing_rewards))) if len(losing_rewards) > 0 else 0.0
            profit_loss_ratio = avg_win / avg_loss if avg_loss > 0 else float('inf')
            
            metrics.update({
                'max_drawdown': max_drawdown,
                'sharpe_ratio': sharpe_ratio,
                'win_rate': win_rate,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'profit_loss_ratio': profit_loss_ratio,
                'final_cumulative_reward': float(cumulative_rewards[-1])
            })
        
        # 稳定性指标
        if len(rewards_array) >= 10:
            # 分段稳定性（前半段vs后半段）
            mid_point = len(rewards_array) // 2
            first_half_mean = float(np.mean(rewards_array[:mid_point]))
            second_half_mean = float(np.mean(rewards_array[mid_point:]))
            stability_ratio = second_half_mean / first_half_mean if first_half_mean != 0 else 1.0
            
            metrics['stability_ratio'] = stability_ratio
        
        return metrics
    
    def summarize_results(self) -> Dict[str, Any]:
        """
        汇总验证结果
        
        Returns:
            Dict[str, Any]: 汇总统计信息
        """
        if not self.validation_results:
            return {"message": "没有可用的验证结果"}
        
        # 提取各折的指标
        test_rewards = []
        validation_rewards = []
        test_metrics = []
        validation_metrics = []
        
        for result in self.validation_results:
            test_result = result['test_result']
            validation_result = result['validation_result']
            
            test_rewards.extend(test_result['episode_rewards'])
            validation_rewards.extend(validation_result['episode_rewards'])
            
            # 计算每折的指标
            test_fold_metrics = self.calculate_metrics(test_result['episode_rewards'])
            validation_fold_metrics = self.calculate_metrics(validation_result['episode_rewards'])
            
            test_metrics.append(test_fold_metrics)
            validation_metrics.append(validation_fold_metrics)
        
        # 汇总指标
        summary = {
            'num_folds': len(self.validation_results),
            'test_summary': self._aggregate_metrics(test_metrics, test_rewards),
            'validation_summary': self._aggregate_metrics(validation_metrics, validation_rewards),
        }
        
        # 稳定性分析
        if len(test_metrics) > 1:
            summary['stability_analysis'] = self._analyze_stability(test_metrics, validation_metrics)
        
        # 时间分析
        summary['temporal_analysis'] = self._analyze_temporal_performance()
        
        self.summary_stats = summary
        return summary
    
    def _aggregate_metrics(self, fold_metrics: List[Dict], all_rewards: List[float]) -> Dict[str, Any]:
        """
        聚合各折的指标
        
        Args:
            fold_metrics: 各折指标列表
            all_rewards: 所有奖励
            
        Returns:
            Dict[str, Any]: 聚合指标
        """
        if not fold_metrics:
            return {}
        
        # 计算各指标的统计量
        aggregated = {}
        metric_names = fold_metrics[0].keys()
        
        for metric_name in metric_names:
            values = [fm.get(metric_name, 0) for fm in fold_metrics if fm.get(metric_name) is not None]
            if values:
                aggregated[f'{metric_name}_mean'] = float(np.mean(values))
                aggregated[f'{metric_name}_std'] = float(np.std(values))
                aggregated[f'{metric_name}_min'] = float(np.min(values))
                aggregated[f'{metric_name}_max'] = float(np.max(values))
        
        # 整体指标
        overall_metrics = self.calculate_metrics(all_rewards)
        aggregated['overall'] = overall_metrics
        
        # 一致性指标
        mean_rewards = [fm.get('mean_reward', 0) for fm in fold_metrics]
        if len(mean_rewards) > 1:
            consistency = 1 - (np.std(mean_rewards) / np.mean(mean_rewards)) if np.mean(mean_rewards) != 0 else 0
            aggregated['consistency_score'] = max(0, float(consistency))
        
        return aggregated
    
    def _analyze_stability(self, test_metrics: List[Dict], validation_metrics: List[Dict]) -> Dict[str, float]:
        """
        分析模型稳定性
        
        Args:
            test_metrics: 测试集指标
            validation_metrics: 验证集指标
            
        Returns:
            Dict[str, float]: 稳定性分析结果
        """
        stability = {}
        
        # 测试集和验证集的一致性
        test_rewards = [m.get('mean_reward', 0) for m in test_metrics]
        val_rewards = [m.get('mean_reward', 0) for m in validation_metrics]
        
        if len(test_rewards) == len(val_rewards) and len(test_rewards) > 0:
            correlation = float(np.corrcoef(test_rewards, val_rewards)[0, 1]) if len(test_rewards) > 1 else 1.0
            stability['test_validation_correlation'] = correlation
            
            # 平均差异
            avg_diff = float(np.mean(np.abs(np.array(test_rewards) - np.array(val_rewards))))
            stability['test_validation_avg_diff'] = avg_diff
        
        # 时间稳定性（前半部分vs后半部分）
        if len(test_rewards) >= 4:
            mid = len(test_rewards) // 2
            early_performance = np.mean(test_rewards[:mid])
            late_performance = np.mean(test_rewards[mid:])
            
            temporal_stability = late_performance / early_performance if early_performance != 0 else 1.0
            stability['temporal_stability'] = float(temporal_stability)
        
        # 方差稳定性
        test_stds = [m.get('std_reward', 0) for m in test_metrics]
        if test_stds:
            stability['avg_volatility'] = float(np.mean(test_stds))
            stability['volatility_consistency'] = float(1 - np.std(test_stds) / np.mean(test_stds)) if np.mean(test_stds) > 0 else 1.0
        
        return stability
    
    def _analyze_temporal_performance(self) -> Dict[str, Any]:
        """
        分析时间维度的性能变化
        
        Returns:
            Dict[str, Any]: 时间分析结果
        """
        if not self.validation_results:
            return {}
        
        temporal = {}
        
        # 提取时间序列数据
        fold_dates = []
        test_performances = []
        validation_performances = []
        
        for result in self.validation_results:
            split_info = result['split_info']
            fold_dates.append(split_info['test_start_date'])
            test_performances.append(result['test_result']['mean_reward'])
            validation_performances.append(result['validation_result']['mean_reward'])
        
        if len(fold_dates) > 1:
            # 性能趋势
            x = np.arange(len(test_performances))
            test_trend = np.polyfit(x, test_performances, 1)[0]  # 斜率
            val_trend = np.polyfit(x, validation_performances, 1)[0]
            
            temporal['test_performance_trend'] = float(test_trend)
            temporal['validation_performance_trend'] = float(val_trend)
            
            # 季节性分析（如果数据跨度足够）
            if len(fold_dates) >= 12:  # 至少一年的数据
                monthly_performance = {}
                for date, perf in zip(fold_dates, test_performances):
                    month = date.month
                    if month not in monthly_performance:
                        monthly_performance[month] = []
                    monthly_performance[month].append(perf)
                
                monthly_avg = {month: np.mean(perfs) for month, perfs in monthly_performance.items()}
                temporal['monthly_performance'] = monthly_avg
        
        temporal['date_range'] = {
            'start': str(fold_dates[0]) if fold_dates else None,
            'end': str(fold_dates[-1]) if fold_dates else None,
            'total_periods': len(fold_dates)
        }
        
        return temporal
    
    def save_results(self, filepath: str, include_detailed: bool = True) -> None:
        """
        保存验证结果到文件
        
        Args:
            filepath: 保存路径
            include_detailed: 是否包含详细的折结果
        """
        try:
            result_data = {
                'timestamp': datetime.now().isoformat(),
                'config': self.config.to_dict(),
                'validation_config': {
                    'train_ratio': self.train_ratio,
                    'test_ratio': self.test_ratio,
                    'validation_ratio': self.validation_ratio,
                    'step_ratio': self.step_ratio,
                    'min_train_size': self.min_train_size
                },
                'summary_statistics': self.summary_stats,
                'num_folds': len(self.validation_results)
            }
            
            if include_detailed:
                # 简化详细结果（移除大型数组以减少文件大小）
                simplified_results = []
                for result in self.validation_results:
                    simplified = {
                        'fold': result['fold'],
                        'split_info': result['split_info'],
                        'fold_duration': result['fold_duration'],
                        'test_metrics': self.calculate_metrics(result['test_result']['episode_rewards']),
                        'validation_metrics': self.calculate_metrics(result['validation_result']['episode_rewards']),
                        'training_summary': {
                            'best_reward': result['training_result'].get('best_reward', 0),
                            'num_iterations': result['training_result'].get('num_iterations', 0),
                            'total_time': result['training_result'].get('total_time', 0)
                        }
                    }
                    simplified_results.append(simplified)
                
                result_data['detailed_results'] = simplified_results
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(result_data, f, indent=2, ensure_ascii=False, default=str)
            
            self.logger.info(f"验证结果已保存: {filepath}")
            
        except Exception as e:
            self.logger.error(f"保存验证结果失败: {e}")
            raise
    
    def get_performance_summary(self) -> str:
        """
        获取性能摘要文本
        
        Returns:
            str: 格式化的性能摘要
        """
        if not self.summary_stats:
            return "没有可用的验证结果"
        
        summary_lines = [
            "=== Walk Forward 验证结果摘要 ===",
            f"验证折数: {self.summary_stats.get('num_folds', 0)}",
            ""
        ]
        
        # 测试集性能
        test_summary = self.summary_stats.get('test_summary', {})
        if test_summary:
            overall = test_summary.get('overall', {})
            summary_lines.extend([
                "测试集性能:",
                f"  平均奖励: {overall.get('mean_reward', 0):.4f} ± {overall.get('std_reward', 0):.4f}",
                f"  夏普比率: {overall.get('sharpe_ratio', 0):.4f}",
                f"  最大回撤: {overall.get('max_drawdown', 0):.4f}",
                f"  胜率: {overall.get('win_rate', 0):.2%}",
                ""
            ])
        
        # 验证集性能
        val_summary = self.summary_stats.get('validation_summary', {})
        if val_summary:
            overall = val_summary.get('overall', {})
            summary_lines.extend([
                "验证集性能:",
                f"  平均奖励: {overall.get('mean_reward', 0):.4f} ± {overall.get('std_reward', 0):.4f}",
                f"  夏普比率: {overall.get('sharpe_ratio', 0):.4f}",
                f"  最大回撤: {overall.get('max_drawdown', 0):.4f}",
                f"  胜率: {overall.get('win_rate', 0):.2%}",
                ""
            ])
        
        # 稳定性分析
        stability = self.summary_stats.get('stability_analysis', {})
        if stability:
            summary_lines.extend([
                "稳定性分析:",
                f"  测试-验证相关性: {stability.get('test_validation_correlation', 0):.4f}",
                f"  时间稳定性: {stability.get('temporal_stability', 1):.4f}",
                f"  一致性评分: {test_summary.get('consistency_score', 0):.4f}",
                ""
            ])
        
        return "\n".join(summary_lines)