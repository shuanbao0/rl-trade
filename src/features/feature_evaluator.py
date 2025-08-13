"""
Experiment #005: 科学特征评估框架 - 项目集成版本
实现严格的特征选择和验证机制，集成到现有特征工程系统
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from scipy import stats
import warnings
import logging
from ..utils.logger import setup_logger, get_default_log_file
from ..utils.config import Config

warnings.filterwarnings('ignore')

@dataclass
class FeatureEvaluationResult:
    """特征评估结果"""
    feature_name: str
    overall_score: float
    individual_scores: Dict[str, float]
    recommendation: str
    p_value: float
    effect_size: float
    confidence_interval: Tuple[float, float]
    evaluation_details: Dict[str, Any]

@dataclass  
class FeatureEvaluatorConfig:
    """特征评估器配置"""
    # 评估权重配置
    evaluation_weights: Dict[str, float] = None
    # 阈值配置
    thresholds: Dict[str, float] = None
    # 金融理论评分
    theory_scores: Dict[str, float] = None
    
    def __post_init__(self):
        if self.evaluation_weights is None:
            self.evaluation_weights = {
                'performance': 0.30,      # 性能改进权重最高
                'significance': 0.20,     # 统计显著性很重要
                'importance': 0.20,       # 特征重要性
                'redundancy': 0.15,       # 冗余度检查
                'cost': 0.10,            # 计算成本
                'theory': 0.05           # 理论支持
            }
        
        if self.thresholds is None:
            self.thresholds = {
                'significance_alpha': 0.05,
                'min_effect_size': 0.2,
                'max_redundancy': 0.8,
                'min_importance': 0.01,
                'acceptance_score': 0.6
            }
        
        if self.theory_scores is None:
            self.theory_scores = {
                # 经典技术分析指标
                'RSI_14': 0.9, 'MACD': 0.9, 'EMA_21': 0.9, 'SMA_20': 0.8,
                'ATR_14': 0.85, 'Williams_R_14': 0.8, 'CCI_20': 0.8,
                'Stochastic_K_14': 0.8, 'BB_Width_20': 0.8,
                
                # 高级技术指标
                'ADX_14': 0.85, 'Parabolic_SAR': 0.7, 'Ichimoku': 0.7,
                'OBV': 0.75, 'MFI': 0.75,
                
                # 波动率指标
                'HV_20': 0.8, 'ATR_Ratio_14_50': 0.75, 'GARCH': 0.7,
                
                # 统计指标
                'Rolling_Mean_20': 0.6, 'Rolling_Std_20': 0.6,
                'Z_Score': 0.6, 'Skewness': 0.5, 'Kurtosis': 0.4,
                
                # 默认分数
                'default': 0.5
            }

class FeatureEvaluator:
    """
    科学的特征评估和选择框架 - Experiment #005
    
    集成到项目特征工程系统中，提供：
    1. 性能改进评估 - 对模型性能的实际贡献
    2. 统计显著性检验 - 改进是否统计显著
    3. 特征重要性分析 - 在模型中的重要程度
    4. 冗余度分析 - 与现有特征的相关性
    5. 计算成本评估 - 计算复杂度分析
    6. 金融理论支持 - 理论基础评分
    """
    
    def __init__(self, config: Optional[Config] = None, 
                 evaluator_config: Optional[FeatureEvaluatorConfig] = None):
        """
        初始化特征评估器
        
        Args:
            config: 项目配置
            evaluator_config: 评估器专用配置
        """
        self.config = config or Config()
        self.evaluator_config = evaluator_config or FeatureEvaluatorConfig()
        
        # 初始化日志
        self.logger = setup_logger(
            name="FeatureEvaluator",
            level="INFO",
            log_file=get_default_log_file("feature_evaluator")
        )
        
        # 评估历史记录
        self.evaluation_history = []
        self.feature_rankings = {}
        
        self.logger.info("FeatureEvaluator (Experiment #005) initialized successfully")
    
    def evaluate_single_feature(self, 
                               feature_name: str,
                               baseline_results: List[Dict],
                               feature_results: List[Dict],
                               feature_data: Optional[pd.Series] = None,
                               existing_features: Optional[pd.DataFrame] = None) -> FeatureEvaluationResult:
        """
        评估单个特征的价值
        
        Args:
            feature_name: 特征名称
            baseline_results: 基准实验结果列表
            feature_results: 加入新特征后的实验结果列表
            feature_data: 特征数据（用于重要性和冗余度分析）
            existing_features: 现有特征数据
        
        Returns:
            FeatureEvaluationResult: 评估结果
        """
        
        self.logger.info(f"开始评估特征: {feature_name}")
        
        # 提取关键指标
        baseline_returns = [r.get('mean_return', 0) for r in baseline_results]
        feature_returns = [r.get('mean_return', 0) for r in feature_results]
        
        # 计算各个评估维度的分数
        scores = {}
        evaluation_details = {}
        
        # 1. 性能改进评分
        performance_result = self._evaluate_performance_improvement(
            baseline_returns, feature_returns
        )
        scores['performance'] = performance_result['score']
        evaluation_details['performance'] = performance_result
        
        # 2. 统计显著性评分
        significance_result = self._evaluate_statistical_significance(
            baseline_returns, feature_returns
        )
        scores['significance'] = significance_result['score']
        p_value = significance_result['p_value']
        effect_size = significance_result['effect_size']
        confidence_interval = significance_result['confidence_interval']
        evaluation_details['significance'] = significance_result
        
        # 3. 特征重要性评分
        importance_result = self._evaluate_feature_importance(
            feature_name, feature_data
        )
        scores['importance'] = importance_result['score']
        evaluation_details['importance'] = importance_result
        
        # 4. 冗余度评分
        redundancy_result = self._evaluate_redundancy(
            feature_data, existing_features
        )
        scores['redundancy'] = redundancy_result['score']
        evaluation_details['redundancy'] = redundancy_result
        
        # 5. 计算成本评分
        cost_result = self._evaluate_computational_cost(feature_name)
        scores['cost'] = cost_result['score']
        evaluation_details['cost'] = cost_result
        
        # 6. 金融理论支持评分
        theory_result = self._evaluate_theory_support(feature_name)
        scores['theory'] = theory_result['score']
        evaluation_details['theory'] = theory_result
        
        # 计算综合评分
        overall_score = sum(
            scores[dimension] * self.evaluator_config.evaluation_weights[dimension]
            for dimension in self.evaluator_config.evaluation_weights
        )
        
        # 生成推荐
        recommendation = self._make_recommendation(
            overall_score, p_value, effect_size
        )
        
        # 创建结果
        result = FeatureEvaluationResult(
            feature_name=feature_name,
            overall_score=overall_score,
            individual_scores=scores,
            recommendation=recommendation,
            p_value=p_value,
            effect_size=effect_size,
            confidence_interval=confidence_interval,
            evaluation_details=evaluation_details
        )
        
        # 记录评估历史
        self.evaluation_history.append(result)
        self.feature_rankings[feature_name] = overall_score
        
        self.logger.info(f"特征 {feature_name} 评估完成: {overall_score:.3f} ({recommendation})")
        
        return result
    
    def _evaluate_performance_improvement(self, baseline: List[float], 
                                        feature: List[float]) -> Dict[str, Any]:
        """评估性能改进"""
        if not baseline or not feature:
            return {'score': 0.0, 'improvement': 0.0, 'baseline_mean': 0.0, 'feature_mean': 0.0}
        
        baseline_mean = np.mean(baseline)
        feature_mean = np.mean(feature)
        
        # 相对改进率
        if baseline_mean != 0:
            improvement = (feature_mean - baseline_mean) / abs(baseline_mean)
        else:
            improvement = feature_mean - baseline_mean
        
        # 转换为0-1分数
        # 改进>20%得满分，改进<-10%得0分
        if improvement >= 0.2:
            score = 1.0
        elif improvement <= -0.1:
            score = 0.0
        else:
            score = (improvement + 0.1) / 0.3
        
        return {
            'score': max(0.0, min(1.0, score)),
            'improvement': improvement,
            'baseline_mean': baseline_mean,
            'feature_mean': feature_mean
        }
    
    def _evaluate_statistical_significance(self, baseline: List[float], 
                                         feature: List[float]) -> Dict:
        """评估统计显著性"""
        if len(baseline) < 3 or len(feature) < 3:
            return {
                'score': 0.0,
                'p_value': 1.0,
                'effect_size': 0.0,
                'confidence_interval': (0.0, 0.0),
                't_statistic': 0.0,
                'test_type': 'insufficient_data'
            }
        
        try:
            # 进行t检验
            t_stat, p_value = stats.ttest_ind(feature, baseline, equal_var=False)
            
            # 计算效应大小 (Cohen's d)
            pooled_std = np.sqrt((np.var(baseline) + np.var(feature)) / 2)
            if pooled_std > 0:
                effect_size = (np.mean(feature) - np.mean(baseline)) / pooled_std
            else:
                effect_size = 0.0
            
            # 计算置信区间
            diff_mean = np.mean(feature) - np.mean(baseline)
            diff_se = np.sqrt(np.var(feature)/len(feature) + np.var(baseline)/len(baseline))
            confidence_interval = (
                diff_mean - 1.96 * diff_se,
                diff_mean + 1.96 * diff_se
            )
            
            # 转换为评分
            if p_value < 0.01:
                score = 1.0
            elif p_value < 0.05:
                score = 0.8
            elif p_value < 0.1:
                score = 0.5
            else:
                score = 0.0
            
            return {
                'score': score,
                'p_value': p_value,
                'effect_size': effect_size,
                'confidence_interval': confidence_interval,
                't_statistic': t_stat,
                'test_type': 'welch_t_test'
            }
        
        except Exception as e:
            self.logger.warning(f"统计显著性检验错误: {e}")
            return {
                'score': 0.0,
                'p_value': 1.0,
                'effect_size': 0.0,
                'confidence_interval': (0.0, 0.0),
                't_statistic': 0.0,
                'test_type': 'error'
            }
    
    def _evaluate_feature_importance(self, feature_name: str, 
                                   feature_data: Optional[pd.Series]) -> Dict[str, Any]:
        """评估特征重要性"""
        if feature_data is None:
            return {'score': 0.5, 'method': 'default', 'cv': None, 'information_gain': None}
        
        try:
            # 计算特征的变异系数作为重要性指标
            if len(feature_data) > 1:
                mean_val = np.mean(feature_data)
                std_val = np.std(feature_data)
                cv = std_val / (abs(mean_val) + 1e-8)
                
                # 计算信息增益代理指标
                # 使用分位数差异作为信息增益的代理
                q75 = np.percentile(feature_data, 75)
                q25 = np.percentile(feature_data, 25)
                iqr = q75 - q25
                information_gain = iqr / (std_val + 1e-8) if std_val > 0 else 0
                
                # 变异系数在0.1-2之间时重要性较高
                if 0.1 <= cv <= 2.0:
                    cv_score = 0.8
                elif cv > 2.0:
                    cv_score = 0.6  # 过于波动可能是噪声
                else:
                    cv_score = 0.4  # 变化太小可能信息量不足
                
                # 信息增益加成
                ig_score = min(1.0, information_gain * 0.5)
                
                final_score = (cv_score * 0.7 + ig_score * 0.3)
                
                return {
                    'score': final_score,
                    'method': 'coefficient_of_variation_with_information_gain',
                    'cv': cv,
                    'information_gain': information_gain,
                    'cv_score': cv_score,
                    'ig_score': ig_score
                }
            else:
                return {'score': 0.5, 'method': 'insufficient_data', 'cv': None, 'information_gain': None}
                
        except Exception as e:
            self.logger.warning(f"特征重要性评估错误: {e}")
            return {'score': 0.5, 'method': 'error', 'cv': None, 'information_gain': None}
    
    def _evaluate_redundancy(self, feature_data: Optional[pd.Series], 
                           existing_features: Optional[pd.DataFrame]) -> Dict[str, Any]:
        """评估特征冗余度"""
        if feature_data is None or existing_features is None:
            return {'score': 0.8, 'max_correlation': 0.0, 'correlated_features': []}
        
        try:
            max_correlation = 0.0
            correlated_features = []
            
            for col in existing_features.columns:
                if len(feature_data) == len(existing_features[col]):
                    # 处理NaN值
                    valid_indices = pd.notna(feature_data) & pd.notna(existing_features[col])
                    if valid_indices.sum() > 10:  # 至少需要10个有效数据点
                        corr = np.corrcoef(
                            feature_data[valid_indices], 
                            existing_features[col][valid_indices]
                        )[0, 1]
                        
                        if np.isfinite(corr):
                            abs_corr = abs(corr)
                            if abs_corr > max_correlation:
                                max_correlation = abs_corr
                            if abs_corr > 0.7:  # 高相关性阈值
                                correlated_features.append((col, corr))
            
            # 相关性越低，冗余度分数越高
            if max_correlation < 0.3:
                score = 1.0
            elif max_correlation < 0.5:
                score = 0.8
            elif max_correlation < 0.7:
                score = 0.5
            else:
                score = 0.2
            
            return {
                'score': score,
                'max_correlation': max_correlation,
                'correlated_features': correlated_features,
                'method': 'pearson_correlation'
            }
            
        except Exception as e:
            self.logger.warning(f"冗余度评估错误: {e}")
            return {'score': 0.8, 'max_correlation': 0.0, 'correlated_features': []}
    
    def _evaluate_computational_cost(self, feature_name: str) -> Dict[str, Any]:
        """评估计算成本"""
        # 简化的计算成本评估
        cost_mapping = {
            # 低成本
            'simple': ['SMA', 'EMA', 'RSI', 'Price', 'Volume', 'Close', 'Open', 'High', 'Low'],
            # 中等成本
            'medium': ['MACD', 'ATR', 'BB', 'Williams_R', 'CCI', 'Stochastic'],
            # 高成本
            'high': ['ADX', 'Parabolic_SAR', 'Ichimoku', 'GARCH', 'Wavelet'],
            # 很高成本
            'very_high': ['FFT', 'Neural', 'Complex', 'Multi_timeframe', 'Cross_correlation']
        }
        
        feature_lower = feature_name.lower()
        cost_level = 'unknown'
        
        for level, keywords in cost_mapping.items():
            if any(keyword.lower() in feature_lower for keyword in keywords):
                cost_level = level
                break
        
        score_mapping = {
            'simple': 1.0,
            'medium': 0.8,
            'high': 0.6,
            'very_high': 0.4,
            'unknown': 0.7
        }
        
        score = score_mapping[cost_level]
        
        return {
            'score': score,
            'cost_level': cost_level,
            'method': 'keyword_based_estimation'
        }
    
    def _evaluate_theory_support(self, feature_name: str) -> Dict[str, Any]:
        """评估金融理论支持"""
        feature_base = feature_name.split('_')[0] if '_' in feature_name else feature_name
        
        score = self.evaluator_config.theory_scores.get(
            feature_name, 
            self.evaluator_config.theory_scores.get(
                feature_base, 
                self.evaluator_config.theory_scores['default']
            )
        )
        
        return {
            'score': score,
            'matched_feature': feature_name if feature_name in self.evaluator_config.theory_scores else feature_base,
            'method': 'theory_score_mapping'
        }
    
    def _make_recommendation(self, score: float, p_value: float, 
                           effect_size: float) -> str:
        """基于评分生成推荐"""
        if score >= 0.8 and p_value < 0.05 and abs(effect_size) > 0.2:
            return "ACCEPT - 强烈推荐使用"
        elif score >= 0.6 and p_value < 0.1:
            return "CONDITIONAL - 谨慎使用，需要进一步验证"
        elif score >= 0.4:
            return "UNCERTAIN - 价值不明确，建议更多测试"
        else:
            return "REJECT - 不推荐使用"
    
    def batch_evaluate_features(self, 
                              candidate_features: Dict[str, str],
                              baseline_results: List[Dict],
                              feature_test_results: Dict[str, List[Dict]],
                              feature_data: Optional[pd.DataFrame] = None) -> List[FeatureEvaluationResult]:
        """
        批量评估候选特征
        
        Args:
            candidate_features: 候选特征映射 {name: description}
            baseline_results: 基准测试结果
            feature_test_results: 每个特征的测试结果 {feature_name: results}
            feature_data: 特征数据
        
        Returns:
            List[FeatureEvaluationResult]: 排序后的评估结果列表
        """
        results = []
        
        self.logger.info(f"开始批量评估 {len(candidate_features)} 个特征")
        
        for feature_name in candidate_features:
            if feature_name in feature_test_results:
                # 获取特征数据
                single_feature_data = None
                if feature_data is not None and feature_name in feature_data.columns:
                    single_feature_data = feature_data[feature_name]
                
                # 获取现有特征数据（排除当前特征）
                existing_features = None
                if feature_data is not None:
                    existing_features = feature_data.drop(columns=[feature_name], errors='ignore')
                
                # 评估特征
                result = self.evaluate_single_feature(
                    feature_name=feature_name,
                    baseline_results=baseline_results,
                    feature_results=feature_test_results[feature_name],
                    feature_data=single_feature_data,
                    existing_features=existing_features
                )
                
                results.append(result)
        
        # 按综合评分排序
        results.sort(key=lambda x: x.overall_score, reverse=True)
        
        self.logger.info(f"批量评估完成，共评估 {len(results)} 个特征")
        
        return results
    
    def select_best_features(self, 
                           evaluation_results: List[FeatureEvaluationResult],
                           max_features: int = 2) -> List[str]:
        """
        从评估结果中选择最佳特征
        
        Args:
            evaluation_results: 特征评估结果列表
            max_features: 最大选择特征数
        
        Returns:
            List[str]: 选择的特征名称列表
        """
        # 过滤出推荐使用的特征
        recommended = [
            result for result in evaluation_results
            if result.recommendation.startswith("ACCEPT") or 
               (result.recommendation.startswith("CONDITIONAL") and 
                result.overall_score >= self.evaluator_config.thresholds['acceptance_score'])
        ]
        
        # 选择前N个最佳特征
        selected = recommended[:max_features]
        
        selected_names = [result.feature_name for result in selected]
        
        self.logger.info(f"选择了 {len(selected_names)} 个最佳特征: {selected_names}")
        
        return selected_names
    
    def generate_evaluation_report(self, 
                                 results: List[FeatureEvaluationResult]) -> str:
        """生成特征评估报告"""
        report = []
        report.append("=" * 80)
        report.append("Experiment #005 特征评估报告")
        report.append("=" * 80)
        report.append(f"评估特征数量: {len(results)}")
        report.append(f"评估时间: {pd.Timestamp.now()}")
        report.append("")
        
        for i, result in enumerate(results, 1):
            report.append(f"{i}. {result.feature_name}")
            report.append(f"   综合评分: {result.overall_score:.3f}")
            report.append(f"   推荐: {result.recommendation}")
            report.append(f"   统计显著性: p={result.p_value:.4f}, effect_size={result.effect_size:.3f}")
            report.append(f"   置信区间: [{result.confidence_interval[0]:.4f}, {result.confidence_interval[1]:.4f}]")
            report.append(f"   各维度评分:")
            
            for dimension, score in result.individual_scores.items():
                report.append(f"     {dimension}: {score:.3f}")
            
            report.append("")
        
        # 汇总统计
        accepted = sum(1 for r in results if r.recommendation.startswith("ACCEPT"))
        conditional = sum(1 for r in results if r.recommendation.startswith("CONDITIONAL"))
        rejected = sum(1 for r in results if r.recommendation.startswith("REJECT"))
        uncertain = sum(1 for r in results if r.recommendation.startswith("UNCERTAIN"))
        
        report.append("汇总统计:")
        report.append(f"  强烈推荐: {accepted} 个特征")
        report.append(f"  条件接受: {conditional} 个特征")
        report.append(f"  价值不明: {uncertain} 个特征")
        report.append(f"  不推荐: {rejected} 个特征")
        
        report.append("")
        report.append("评估方法:")
        report.append("  1. 性能改进评估 - 基于实际交易回报提升")
        report.append("  2. 统计显著性检验 - Welch t检验")
        report.append("  3. 特征重要性分析 - 变异系数与信息增益")
        report.append("  4. 冗余度分析 - Pearson相关性检验")
        report.append("  5. 计算成本评估 - 基于特征复杂度")
        report.append("  6. 金融理论支持 - 基于理论基础评分")
        
        return "\n".join(report)
    
    def get_feature_rankings(self) -> Dict[str, float]:
        """获取特征排名"""
        return dict(sorted(self.feature_rankings.items(), key=lambda x: x[1], reverse=True))
    
    def get_evaluation_summary(self) -> Dict[str, Any]:
        """获取评估摘要"""
        if not self.evaluation_history:
            return {"status": "no_evaluations_performed"}
        
        scores = [r.overall_score for r in self.evaluation_history]
        recommendations = [r.recommendation for r in self.evaluation_history]
        
        return {
            "total_evaluations": len(self.evaluation_history),
            "mean_score": np.mean(scores),
            "median_score": np.median(scores),
            "std_score": np.std(scores),
            "score_range": [min(scores), max(scores)],
            "recommendation_counts": {
                "ACCEPT": sum(1 for r in recommendations if r.startswith("ACCEPT")),
                "CONDITIONAL": sum(1 for r in recommendations if r.startswith("CONDITIONAL")),
                "UNCERTAIN": sum(1 for r in recommendations if r.startswith("UNCERTAIN")),
                "REJECT": sum(1 for r in recommendations if r.startswith("REJECT"))
            },
            "top_features": list(self.get_feature_rankings().keys())[:5]
        }

# 工厂函数
def create_feature_evaluator(config: Optional[Config] = None,
                           evaluator_config: Optional[FeatureEvaluatorConfig] = None) -> FeatureEvaluator:
    """创建特征评估器的工厂方法"""
    return FeatureEvaluator(config=config, evaluator_config=evaluator_config)

if __name__ == "__main__":
    # 测试特征评估器
    print("🧪 测试FeatureEvaluator (Experiment #005)...")
    
    evaluator = create_feature_evaluator()
    
    # 模拟测试数据
    baseline_results = [
        {'mean_return': -0.25, 'sharpe_ratio': -0.5},
        {'mean_return': -0.23, 'sharpe_ratio': -0.4},
        {'mean_return': -0.27, 'sharpe_ratio': -0.6}
    ]
    
    feature_results = [
        {'mean_return': -0.18, 'sharpe_ratio': -0.2},
        {'mean_return': -0.15, 'sharpe_ratio': -0.1},
        {'mean_return': -0.20, 'sharpe_ratio': -0.3}
    ]
    
    # 评估单个特征
    result = evaluator.evaluate_single_feature(
        feature_name="Williams_R_14",
        baseline_results=baseline_results,
        feature_results=feature_results
    )
    
    print(f"SUCCESS: 特征评估完成")
    print(f"   特征名称: {result.feature_name}")
    print(f"   综合评分: {result.overall_score:.3f}")
    print(f"   推荐: {result.recommendation}")
    print(f"   p值: {result.p_value:.4f}")
    print(f"   效应大小: {result.effect_size:.3f}")
    
    # 获取评估摘要
    summary = evaluator.get_evaluation_summary()
    print(f"   评估摘要: {summary['total_evaluations']} 个特征已评估")
    
    print("\n🎯 FeatureEvaluator (Experiment #005) 准备就绪!")