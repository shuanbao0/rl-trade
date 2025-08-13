"""
Time Series Validator
时间序列严格验证器

Purpose: 实现专门用于时间序列数据的严格样本外验证
解决历史实验中过拟合和泛化能力差的问题

Key Features:
- 时间感知的数据分割，防止未来数据泄露
- 滚动窗口验证 (Walk-Forward Analysis)
- 蒙特卡洛验证确保结果稳定性
- 样本外性能监控和预警
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Callable
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from abc import ABC, abstractmethod

from ..utils.logger import setup_logger


@dataclass
class ValidationResult:
    """验证结果数据类"""
    train_performance: Dict[str, float]
    validation_performance: Dict[str, float]
    test_performance: Dict[str, float]
    correlation_metrics: Dict[str, float]
    overfitting_score: float
    stability_score: float
    validation_passed: bool
    warnings: List[str]
    timestamp: str


class TimeSeriesValidator:
    """
    专门用于强化学习交易系统的时间序列验证器
    
    实现严格的样本外验证，防止未来数据泄露和过拟合
    解决历史实验中训练收敛但评估失败的问题
    """
    
    def __init__(self, config=None):
        self.config = config or {}
        self.logger = setup_logger("TimeSeriesValidator")
        
        # 验证配置参数
        self.train_ratio = self.config.get('train_ratio', 0.6)      # 60%训练
        self.validation_ratio = self.config.get('validation_ratio', 0.2)  # 20%验证
        self.test_ratio = self.config.get('test_ratio', 0.2)        # 20%测试
        
        # 过拟合检测阈值
        self.overfitting_threshold = self.config.get('overfitting_threshold', 0.2)  # 20%性能差距
        self.correlation_threshold = self.config.get('correlation_threshold', 0.8)   # 最小相关性要求
        
        # 稳定性验证参数
        self.min_validation_episodes = self.config.get('min_validation_episodes', 10)
        self.stability_runs = self.config.get('stability_runs', 5)
        
        self.logger.info("TimeSeriesValidator初始化完成")
        self.logger.info(f"数据分割比例 - 训练:{self.train_ratio}, 验证:{self.validation_ratio}, 测试:{self.test_ratio}")
    
    def create_time_aware_splits(self, data: pd.DataFrame) -> Dict[str, Tuple[int, int]]:
        """
        创建时间感知的数据分割
        严格按照时间顺序分割，防止未来数据泄露
        
        Args:
            data: 时间序列数据
            
        Returns:
            splits: 包含各阶段起止索引的字典
        """
        total_length = len(data)
        
        # 计算分割点
        train_end = int(total_length * self.train_ratio)
        val_end = int(total_length * (self.train_ratio + self.validation_ratio))
        
        splits = {
            'train': (0, train_end),
            'validation': (train_end, val_end),
            'test': (val_end, total_length),
            
            # 扩展信息
            'train_size': train_end,
            'validation_size': val_end - train_end,
            'test_size': total_length - val_end,
            'total_size': total_length
        }
        
        self.logger.info("时间序列数据分割完成:")
        self.logger.info(f"  训练集: {splits['train'][0]}:{splits['train'][1]} ({splits['train_size']}条)")
        self.logger.info(f"  验证集: {splits['validation'][0]}:{splits['validation'][1]} ({splits['validation_size']}条)")
        self.logger.info(f"  测试集: {splits['test'][0]}:{splits['test'][1]} ({splits['test_size']}条)")
        
        return splits
    
    def walk_forward_validation(self, data: pd.DataFrame, model_trainer: Any, 
                              window_size: int = 2000, step_size: int = 500) -> List[ValidationResult]:
        """
        滚动窗口验证 (Walk-Forward Analysis)
        
        Args:
            data: 时间序列数据
            model_trainer: 模型训练器实例
            window_size: 训练窗口大小
            step_size: 滚动步长
            
        Returns:
            results: 每个窗口的验证结果列表
        """
        
        results = []
        total_windows = (len(data) - window_size) // step_size
        
        self.logger.info(f"开始滚动窗口验证: 窗口大小={window_size}, 步长={step_size}, 总窗口数={total_windows}")
        
        for i in range(0, len(data) - window_size, step_size):
            window_start = i
            window_end = i + window_size
            
            # 时间窗口分割
            train_end = window_start + int(window_size * 0.7)
            val_end = window_start + int(window_size * 0.85)
            test_end = window_end
            
            try:
                # 提取各阶段数据
                train_data = data.iloc[window_start:train_end]
                val_data = data.iloc[train_end:val_end]
                test_data = data.iloc[val_end:test_end]
                
                # 训练模型
                self.logger.debug(f"训练窗口 {i//step_size + 1}/{total_windows}: {window_start}-{window_end}")
                model, train_metrics = model_trainer.train_on_window(train_data)
                
                # 验证和测试
                val_metrics = model_trainer.evaluate_on_window(model, val_data)
                test_metrics = model_trainer.evaluate_on_window(model, test_data)
                
                # 计算相关性指标
                correlation_metrics = self._calculate_correlation_metrics(
                    train_metrics, val_metrics, test_metrics
                )
                
                # 计算过拟合和稳定性评分
                overfitting_score = self._calculate_overfitting_score(train_metrics, val_metrics)
                stability_score = self._calculate_stability_score([train_metrics, val_metrics, test_metrics])
                
                # 创建验证结果
                result = ValidationResult(
                    train_performance=train_metrics,
                    validation_performance=val_metrics,
                    test_performance=test_metrics,
                    correlation_metrics=correlation_metrics,
                    overfitting_score=overfitting_score,
                    stability_score=stability_score,
                    validation_passed=self._check_validation_criteria(
                        train_metrics, val_metrics, test_metrics, correlation_metrics
                    ),
                    warnings=self._generate_warnings(train_metrics, val_metrics, correlation_metrics),
                    timestamp=datetime.now().isoformat()
                )
                
                results.append(result)
                
            except Exception as e:
                self.logger.error(f"滚动窗口 {i//step_size + 1} 验证失败: {e}")
                continue
        
        self.logger.info(f"滚动窗口验证完成，成功验证 {len(results)}/{total_windows} 个窗口")
        
        return results
    
    def monte_carlo_validation(self, data: pd.DataFrame, model_trainer: Any, 
                             n_runs: int = 10) -> List[ValidationResult]:
        """
        蒙特卡洛验证 - 多次随机验证确保结果稳定性
        
        Args:
            data: 时间序列数据
            model_trainer: 模型训练器实例
            n_runs: 验证运行次数
            
        Returns:
            results: 所有运行的验证结果列表
        """
        
        results = []
        
        self.logger.info(f"开始蒙特卡洛验证，运行次数: {n_runs}")
        
        for run in range(n_runs):
            try:
                # 设置随机种子确保可重现
                np.random.seed(42 + run)
                
                # 创建时间感知分割
                splits = self.create_time_aware_splits(data)
                
                # 随机选择训练起始点（但保持时间序列顺序）
                max_start_offset = min(100, splits['train_size'] // 10)  # 最多偏移10%
                start_offset = np.random.randint(0, max_start_offset)
                
                # 调整数据分割
                train_start = start_offset
                train_end = splits['train'][1]
                val_start = train_end
                val_end = splits['validation'][1]
                test_start = val_end
                test_end = splits['test'][1]
                
                # 提取数据
                train_data = data.iloc[train_start:train_end]
                val_data = data.iloc[val_start:val_end]
                test_data = data.iloc[test_start:test_end]
                
                self.logger.debug(f"蒙特卡洛运行 {run+1}/{n_runs}, 种子: {42+run}")
                
                # 训练模型
                model, train_metrics = model_trainer.train_with_validation(
                    train_data, val_data, seed=42+run
                )
                
                # 测试模型
                test_metrics = model_trainer.evaluate_model(model, test_data)
                
                # 重新评估验证集（防止训练时的偏差）
                val_metrics = model_trainer.evaluate_model(model, val_data)
                
                # 计算指标
                correlation_metrics = self._calculate_correlation_metrics(
                    train_metrics, val_metrics, test_metrics
                )
                overfitting_score = self._calculate_overfitting_score(train_metrics, val_metrics)
                stability_score = self._calculate_stability_score([train_metrics, val_metrics, test_metrics])
                
                # 创建结果
                result = ValidationResult(
                    train_performance=train_metrics,
                    validation_performance=val_metrics,
                    test_performance=test_metrics,
                    correlation_metrics=correlation_metrics,
                    overfitting_score=overfitting_score,
                    stability_score=stability_score,
                    validation_passed=self._check_validation_criteria(
                        train_metrics, val_metrics, test_metrics, correlation_metrics
                    ),
                    warnings=self._generate_warnings(train_metrics, val_metrics, correlation_metrics),
                    timestamp=datetime.now().isoformat()
                )
                
                results.append(result)
                
            except Exception as e:
                self.logger.error(f"蒙特卡洛运行 {run+1} 失败: {e}")
                continue
        
        self.logger.info(f"蒙特卡洛验证完成，成功运行 {len(results)}/{n_runs} 次")
        
        return results
    
    def validate_model_performance(self, data: pd.DataFrame, model_trainer: Any,
                                 validation_type: str = 'comprehensive') -> Dict[str, Any]:
        """
        全面的模型性能验证
        
        Args:
            data: 时间序列数据
            model_trainer: 模型训练器实例
            validation_type: 验证类型 ('quick', 'standard', 'comprehensive')
            
        Returns:
            validation_report: 完整的验证报告
        """
        
        self.logger.info(f"开始{validation_type}验证")
        
        validation_report = {
            'validation_type': validation_type,
            'start_time': datetime.now().isoformat(),
            'data_info': {
                'total_samples': len(data),
                'date_range': f"{data.index[0]} to {data.index[-1]}",
                'features': list(data.columns)
            }
        }
        
        try:
            if validation_type in ['quick', 'standard', 'comprehensive']:
                # 基础时间序列验证
                splits = self.create_time_aware_splits(data)
                
                train_data = data.iloc[splits['train'][0]:splits['train'][1]]
                val_data = data.iloc[splits['validation'][0]:splits['validation'][1]]
                test_data = data.iloc[splits['test'][0]:splits['test'][1]]
                
                # 训练和评估
                model, train_metrics = model_trainer.train_with_validation(train_data, val_data)
                test_metrics = model_trainer.evaluate_model(model, test_data)
                val_metrics = model_trainer.evaluate_model(model, val_data)
                
                validation_report['basic_validation'] = {
                    'train_metrics': train_metrics,
                    'validation_metrics': val_metrics,
                    'test_metrics': test_metrics
                }
            
            if validation_type in ['standard', 'comprehensive']:
                # 蒙特卡洛验证
                mc_results = self.monte_carlo_validation(data, model_trainer, n_runs=5)
                validation_report['monte_carlo_validation'] = self._summarize_mc_results(mc_results)
            
            if validation_type == 'comprehensive':
                # 滚动窗口验证
                wf_results = self.walk_forward_validation(
                    data, model_trainer, window_size=1500, step_size=300
                )
                validation_report['walk_forward_validation'] = self._summarize_wf_results(wf_results)
            
            # 综合评估
            validation_report['overall_assessment'] = self._generate_overall_assessment(validation_report)
            validation_report['recommendations'] = self._generate_recommendations(validation_report)
            
        except Exception as e:
            self.logger.error(f"验证过程失败: {e}")
            validation_report['error'] = str(e)
            validation_report['validation_passed'] = False
        
        validation_report['end_time'] = datetime.now().isoformat()
        validation_report['duration'] = str(datetime.fromisoformat(validation_report['end_time']) - 
                                           datetime.fromisoformat(validation_report['start_time']))
        
        self.logger.info(f"{validation_type}验证完成")
        
        return validation_report
    
    def _calculate_correlation_metrics(self, train_metrics: Dict, val_metrics: Dict, 
                                     test_metrics: Dict) -> Dict[str, float]:
        """计算相关性指标"""
        correlation_metrics = {}
        
        # 奖励-回报相关性
        for phase, metrics in [('train', train_metrics), ('val', val_metrics), ('test', test_metrics)]:
            if 'episode_rewards' in metrics and 'episode_returns' in metrics:
                rewards = metrics['episode_rewards']
                returns = metrics['episode_returns']
                
                if len(rewards) > 1 and len(returns) > 1:
                    correlation = np.corrcoef(rewards, returns)[0, 1]
                    correlation_metrics[f'{phase}_reward_return_correlation'] = correlation if not np.isnan(correlation) else 0.0
                else:
                    correlation_metrics[f'{phase}_reward_return_correlation'] = 0.0
        
        return correlation_metrics
    
    def _calculate_overfitting_score(self, train_metrics: Dict, val_metrics: Dict) -> float:
        """计算过拟合评分"""
        try:
            # 比较训练和验证性能
            train_return = train_metrics.get('mean_return', 0)
            val_return = val_metrics.get('mean_return', 0)
            
            if abs(train_return) < 1e-10:  # 避免除零
                return 1.0  # 最高过拟合分数
            
            performance_gap = abs(train_return - val_return) / abs(train_return)
            return min(performance_gap, 1.0)  # 限制在[0,1]范围
            
        except Exception:
            return 1.0  # 计算失败时返回最高风险分数
    
    def _calculate_stability_score(self, metrics_list: List[Dict]) -> float:
        """计算稳定性评分"""
        try:
            returns = []
            for metrics in metrics_list:
                if 'mean_return' in metrics:
                    returns.append(metrics['mean_return'])
            
            if len(returns) < 2:
                return 0.0
            
            # 计算收益率的变异系数
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            
            if abs(mean_return) < 1e-10:
                return 0.0
            
            cv = std_return / abs(mean_return)  # 变异系数
            stability_score = max(0, 1 - cv)  # 转换为稳定性分数
            
            return stability_score
            
        except Exception:
            return 0.0
    
    def _check_validation_criteria(self, train_metrics: Dict, val_metrics: Dict, 
                                 test_metrics: Dict, correlation_metrics: Dict) -> bool:
        """检查验证标准是否通过"""
        
        # 1. 奖励-回报相关性检查
        val_correlation = correlation_metrics.get('val_reward_return_correlation', 0)
        if val_correlation < self.correlation_threshold:
            return False
        
        # 2. 过拟合检查
        overfitting_score = self._calculate_overfitting_score(train_metrics, val_metrics)
        if overfitting_score > self.overfitting_threshold:
            return False
        
        # 3. 基本性能检查
        val_return = val_metrics.get('mean_return', -100)
        if val_return < -80:  # 极差性能直接失败
            return False
        
        return True
    
    def _generate_warnings(self, train_metrics: Dict, val_metrics: Dict, 
                         correlation_metrics: Dict) -> List[str]:
        """生成警告信息"""
        warnings = []
        
        # 相关性警告
        val_correlation = correlation_metrics.get('val_reward_return_correlation', 0)
        if val_correlation < self.correlation_threshold:
            warnings.append(f"验证集奖励-回报相关性过低: {val_correlation:.3f} < {self.correlation_threshold}")
        
        # 过拟合警告
        overfitting_score = self._calculate_overfitting_score(train_metrics, val_metrics)
        if overfitting_score > self.overfitting_threshold:
            warnings.append(f"检测到过拟合风险: {overfitting_score:.3f} > {self.overfitting_threshold}")
        
        # 性能警告
        val_return = val_metrics.get('mean_return', 0)
        if val_return < -50:
            warnings.append(f"验证性能极差: 平均回报 {val_return:.1f}%")
        
        return warnings
    
    def _summarize_mc_results(self, mc_results: List[ValidationResult]) -> Dict:
        """总结蒙特卡洛验证结果"""
        if not mc_results:
            return {'error': '无有效的蒙特卡洛验证结果'}
        
        # 提取关键指标
        val_returns = [r.validation_performance.get('mean_return', 0) for r in mc_results]
        test_returns = [r.test_performance.get('mean_return', 0) for r in mc_results]
        correlations = [r.correlation_metrics.get('val_reward_return_correlation', 0) for r in mc_results]
        stability_scores = [r.stability_score for r in mc_results]
        
        passed_runs = sum(1 for r in mc_results if r.validation_passed)
        
        return {
            'total_runs': len(mc_results),
            'passed_runs': passed_runs,
            'pass_rate': passed_runs / len(mc_results),
            'validation_return_stats': {
                'mean': np.mean(val_returns),
                'std': np.std(val_returns),
                'min': np.min(val_returns),
                'max': np.max(val_returns)
            },
            'test_return_stats': {
                'mean': np.mean(test_returns),
                'std': np.std(test_returns),
                'min': np.min(test_returns),
                'max': np.max(test_returns)
            },
            'correlation_stats': {
                'mean': np.mean(correlations),
                'std': np.std(correlations),
                'min': np.min(correlations)
            },
            'stability_stats': {
                'mean': np.mean(stability_scores),
                'std': np.std(stability_scores)
            }
        }
    
    def _summarize_wf_results(self, wf_results: List[ValidationResult]) -> Dict:
        """总结滚动窗口验证结果"""
        if not wf_results:
            return {'error': '无有效的滚动窗口验证结果'}
        
        test_returns = [r.test_performance.get('mean_return', 0) for r in wf_results]
        correlations = [r.correlation_metrics.get('test_reward_return_correlation', 0) for r in wf_results]
        passed_windows = sum(1 for r in wf_results if r.validation_passed)
        
        return {
            'total_windows': len(wf_results),
            'passed_windows': passed_windows,
            'pass_rate': passed_windows / len(wf_results),
            'test_return_trend': {
                'mean': np.mean(test_returns),
                'trend': 'improving' if test_returns[-5:] > test_returns[:5] else 'declining',
                'stability': np.std(test_returns)
            },
            'correlation_trend': {
                'mean': np.mean(correlations),
                'stability': np.std(correlations)
            }
        }
    
    def _generate_overall_assessment(self, validation_report: Dict) -> Dict:
        """生成整体评估"""
        assessment = {
            'validation_status': 'UNKNOWN',
            'confidence_level': 'LOW',
            'key_findings': [],
            'major_concerns': []
        }
        
        # 基于基础验证结果
        if 'basic_validation' in validation_report:
            basic = validation_report['basic_validation']
            test_return = basic['test_metrics'].get('mean_return', -100)
            
            if test_return > -20:
                assessment['key_findings'].append('测试集表现相对可接受')
            elif test_return < -60:
                assessment['major_concerns'].append('测试集表现极差')
        
        # 基于蒙特卡洛结果
        if 'monte_carlo_validation' in validation_report:
            mc = validation_report['monte_carlo_validation']
            if mc.get('pass_rate', 0) > 0.6:
                assessment['key_findings'].append('多次验证显示模型相对稳定')
                assessment['confidence_level'] = 'MEDIUM'
            else:
                assessment['major_concerns'].append('多次验证显示模型不稳定')
        
        # 确定整体状态
        if len(assessment['major_concerns']) == 0:
            assessment['validation_status'] = 'PASSED'
            assessment['confidence_level'] = 'HIGH'
        elif len(assessment['major_concerns']) <= 2:
            assessment['validation_status'] = 'WARNING'
        else:
            assessment['validation_status'] = 'FAILED'
        
        return assessment
    
    def _generate_recommendations(self, validation_report: Dict) -> List[str]:
        """生成改进建议"""
        recommendations = []
        
        assessment = validation_report.get('overall_assessment', {})
        
        if assessment.get('validation_status') == 'FAILED':
            recommendations.append("建议重新审视模型架构和奖励函数设计")
            recommendations.append("考虑减少特征复杂度或增加训练数据")
        
        if 'monte_carlo_validation' in validation_report:
            mc = validation_report['monte_carlo_validation']
            if mc.get('pass_rate', 0) < 0.5:
                recommendations.append("模型稳定性不足，建议调整超参数或增加正则化")
        
        if len(recommendations) == 0:
            recommendations.append("验证结果良好，可以继续模型优化和部署准备")
        
        return recommendations


# 测试和使用示例
if __name__ == "__main__":
    # 创建测试数据
    dates = pd.date_range('2023-01-01', periods=5000, freq='H')
    test_data = pd.DataFrame({
        'price': np.random.randn(5000).cumsum() + 100,
        'volume': np.random.randint(100, 1000, 5000),
        'feature1': np.random.randn(5000),
        'feature2': np.random.randn(5000),
        'feature3': np.random.randn(5000)
    }, index=dates)
    
    # 创建验证器
    validator = TimeSeriesValidator()
    
    # 测试数据分割
    splits = validator.create_time_aware_splits(test_data)
    print("数据分割测试:")
    for key, value in splits.items():
        print(f"  {key}: {value}")
    
    # 测试相关性计算
    mock_train_metrics = {
        'mean_return': -10.5,
        'episode_rewards': [1, 2, 3, 4, 5],
        'episode_returns': [-1, -2, -3, -4, -5]
    }
    mock_val_metrics = {
        'mean_return': -12.3,
        'episode_rewards': [0.8, 1.9, 2.8, 3.9, 4.8],
        'episode_returns': [-1.1, -2.1, -3.1, -4.1, -5.1]
    }
    
    correlation_metrics = validator._calculate_correlation_metrics(
        mock_train_metrics, mock_val_metrics, mock_val_metrics
    )
    print(f"\n相关性计算测试: {correlation_metrics}")
    
    overfitting_score = validator._calculate_overfitting_score(mock_train_metrics, mock_val_metrics)
    print(f"过拟合评分测试: {overfitting_score:.3f}")
    
    print("\nTimeSeriesValidator测试完成")