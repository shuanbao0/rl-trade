"""
FeatureEvaluator 测试文件
测试集成到项目中的科学特征评估框架
"""

import unittest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.features.feature_evaluator import (
    FeatureEvaluator,
    FeatureEvaluatorConfig,
    FeatureEvaluationResult,
    create_feature_evaluator
)
from src.utils.config import Config


class TestFeatureEvaluatorConfig(unittest.TestCase):
    """测试FeatureEvaluatorConfig配置类"""
    
    def test_default_config(self):
        """测试默认配置"""
        config = FeatureEvaluatorConfig()
        
        # 验证评估权重
        self.assertEqual(config.evaluation_weights['performance'], 0.30)
        self.assertEqual(config.evaluation_weights['significance'], 0.20)
        self.assertEqual(config.evaluation_weights['importance'], 0.20)
        self.assertEqual(config.evaluation_weights['redundancy'], 0.15)
        self.assertEqual(config.evaluation_weights['cost'], 0.10)
        self.assertEqual(config.evaluation_weights['theory'], 0.05)
        
        # 验证权重总和为1
        total_weight = sum(config.evaluation_weights.values())
        self.assertAlmostEqual(total_weight, 1.0, places=5)
        
        # 验证阈值
        self.assertEqual(config.thresholds['significance_alpha'], 0.05)
        self.assertEqual(config.thresholds['min_effect_size'], 0.2)
        self.assertEqual(config.thresholds['acceptance_score'], 0.6)
        
        # 验证理论评分
        self.assertIn('RSI_14', config.theory_scores)
        self.assertEqual(config.theory_scores['default'], 0.5)


class TestFeatureEvaluator(unittest.TestCase):
    """测试FeatureEvaluator特征评估器"""
    
    def setUp(self):
        """测试前设置"""
        self.config = Config()
        self.evaluator_config = FeatureEvaluatorConfig()
        self.evaluator = FeatureEvaluator(
            config=self.config,
            evaluator_config=self.evaluator_config
        )
    
    def test_initialization(self):
        """测试初始化"""
        self.assertIsInstance(self.evaluator.config, Config)
        self.assertIsInstance(self.evaluator.evaluator_config, FeatureEvaluatorConfig)
        self.assertEqual(len(self.evaluator.evaluation_history), 0)
        self.assertEqual(len(self.evaluator.feature_rankings), 0)
    
    def test_evaluate_single_feature_basic(self):
        """测试单个特征评估基础功能"""
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
        
        result = self.evaluator.evaluate_single_feature(
            feature_name="RSI_14",
            baseline_results=baseline_results,
            feature_results=feature_results
        )
        
        self.assertIsInstance(result, FeatureEvaluationResult)
        self.assertEqual(result.feature_name, "RSI_14")
        self.assertIsInstance(result.overall_score, float)
        self.assertIsInstance(result.individual_scores, dict)
        self.assertIn(result.recommendation, [
            "ACCEPT - 强烈推荐使用",
            "CONDITIONAL - 谨慎使用，需要进一步验证", 
            "UNCERTAIN - 价值不明确，建议更多测试",
            "REJECT - 不推荐使用"
        ])
        
        # 验证评估历史更新
        self.assertEqual(len(self.evaluator.evaluation_history), 1)
        self.assertIn("RSI_14", self.evaluator.feature_rankings)
    
    def test_performance_improvement_evaluation(self):
        """测试性能改进评估"""
        baseline = [0.1, 0.2, 0.15]
        feature = [0.3, 0.4, 0.35]  # 明显改进
        
        result = self.evaluator._evaluate_performance_improvement(baseline, feature)
        self.assertGreater(result['score'], 0.5)
        self.assertGreater(result['improvement'], 0)
        
        # 测试性能下降
        worse_feature = [0.05, 0.08, 0.06]
        result = self.evaluator._evaluate_performance_improvement(baseline, worse_feature)
        self.assertLess(result['score'], 0.5)
        self.assertLess(result['improvement'], 0)
    
    def test_statistical_significance_evaluation(self):
        """测试统计显著性评估"""
        baseline = [0.1, 0.12, 0.09, 0.11, 0.13]
        feature = [0.2, 0.22, 0.19, 0.21, 0.23]  # 显著改进
        
        result = self.evaluator._evaluate_statistical_significance(baseline, feature)
        self.assertIn('score', result)
        self.assertIn('p_value', result)
        self.assertIn('effect_size', result)
        self.assertIn('confidence_interval', result)
        
        # p值应该很小（显著差异）
        self.assertLess(result['p_value'], 0.1)
        
        # 测试数据不足
        small_baseline = [0.1]
        small_feature = [0.2]
        result = self.evaluator._evaluate_statistical_significance(small_baseline, small_feature)
        self.assertEqual(result['score'], 0.0)
        self.assertEqual(result['p_value'], 1.0)
    
    def test_feature_importance_evaluation(self):
        """测试特征重要性评估"""
        # 创建有变化的特征数据
        feature_data = pd.Series([1.0, 2.0, 1.5, 3.0, 2.5, 1.8, 2.2])
        result = self.evaluator._evaluate_feature_importance("test_feature", feature_data)
        
        self.assertIn('score', result)
        self.assertIn('method', result)
        self.assertGreater(result['score'], 0)
        
        # 测试None数据
        result = self.evaluator._evaluate_feature_importance("test_feature", None)
        self.assertEqual(result['score'], 0.5)
    
    def test_redundancy_evaluation(self):
        """测试冗余度评估"""
        # 创建特征数据
        feature_data = pd.Series([1, 2, 3, 4, 5])
        
        # 创建现有特征（高相关性）
        existing_features = pd.DataFrame({
            'existing_1': [1.1, 2.1, 3.1, 4.1, 5.1],  # 高相关
            'existing_2': [5, 4, 3, 2, 1]  # 负相关
        })
        
        result = self.evaluator._evaluate_redundancy(feature_data, existing_features)
        
        self.assertIn('score', result)
        self.assertIn('max_correlation', result)
        self.assertIn('correlated_features', result)
        # 确保返回值在合理范围内
        self.assertGreaterEqual(result['max_correlation'], 0.0)
        self.assertLessEqual(result['max_correlation'], 1.0)
        self.assertGreaterEqual(result['score'], 0.0)
        self.assertLessEqual(result['score'], 1.0)
        
        # 测试None数据
        result = self.evaluator._evaluate_redundancy(None, None)
        self.assertEqual(result['score'], 0.8)
    
    def test_computational_cost_evaluation(self):
        """测试计算成本评估"""
        # 测试简单指标
        result = self.evaluator._evaluate_computational_cost("SMA_20")
        self.assertEqual(result['cost_level'], 'simple')
        self.assertEqual(result['score'], 1.0)
        
        # 测试复杂指标
        result = self.evaluator._evaluate_computational_cost("ADX_14")
        self.assertEqual(result['cost_level'], 'high')
        self.assertEqual(result['score'], 0.6)
        
        # 测试未知指标
        result = self.evaluator._evaluate_computational_cost("unknown_feature")
        self.assertEqual(result['cost_level'], 'unknown')
        self.assertEqual(result['score'], 0.7)
    
    def test_theory_support_evaluation(self):
        """测试理论支持评估"""
        # 测试已知指标
        result = self.evaluator._evaluate_theory_support("RSI_14")
        # 检查是否是具体的RSI_14分数或默认分数
        expected_score = self.evaluator.evaluator_config.theory_scores.get('RSI_14', 0.5)
        self.assertEqual(result['score'], expected_score)
        
        # 测试基础名称匹配
        result = self.evaluator._evaluate_theory_support("RSI_21")  # 基础名RSI
        # 应该使用RSI的分数或默认分数
        expected_score = self.evaluator.evaluator_config.theory_scores.get('RSI_21', 
                        self.evaluator.evaluator_config.theory_scores.get('RSI', 0.5))
        self.assertEqual(result['score'], expected_score)
        
        # 测试未知指标
        result = self.evaluator._evaluate_theory_support("unknown_indicator")
        self.assertEqual(result['score'], 0.5)  # 默认分数
    
    def test_batch_evaluate_features(self):
        """测试批量特征评估"""
        candidate_features = {
            'RSI_14': 'Relative Strength Index',
            'MACD': 'Moving Average Convergence Divergence'
        }
        
        baseline_results = [
            {'mean_return': -0.1, 'sharpe_ratio': -0.2}
        ] * 3
        
        feature_test_results = {
            'RSI_14': [{'mean_return': 0.1, 'sharpe_ratio': 0.2}] * 3,
            'MACD': [{'mean_return': 0.05, 'sharpe_ratio': 0.1}] * 3
        }
        
        results = self.evaluator.batch_evaluate_features(
            candidate_features=candidate_features,
            baseline_results=baseline_results,
            feature_test_results=feature_test_results
        )
        
        self.assertEqual(len(results), 2)
        self.assertIsInstance(results[0], FeatureEvaluationResult)
        
        # 结果应该按评分排序
        self.assertGreaterEqual(results[0].overall_score, results[1].overall_score)
    
    def test_select_best_features(self):
        """测试最佳特征选择"""
        # 创建模拟评估结果
        results = [
            FeatureEvaluationResult(
                feature_name="RSI_14",
                overall_score=0.85,
                individual_scores={},
                recommendation="ACCEPT - 强烈推荐使用",
                p_value=0.01,
                effect_size=0.5,
                confidence_interval=(0.1, 0.3),
                evaluation_details={}
            ),
            FeatureEvaluationResult(
                feature_name="MACD",
                overall_score=0.65,
                individual_scores={},
                recommendation="CONDITIONAL - 谨慎使用，需要进一步验证",
                p_value=0.08,
                effect_size=0.3,
                confidence_interval=(0.05, 0.15),
                evaluation_details={}
            ),
            FeatureEvaluationResult(
                feature_name="Poor_Feature",
                overall_score=0.3,
                individual_scores={},
                recommendation="REJECT - 不推荐使用",
                p_value=0.5,
                effect_size=0.1,
                confidence_interval=(-0.1, 0.1),
                evaluation_details={}
            )
        ]
        
        selected = self.evaluator.select_best_features(results, max_features=2)
        
        self.assertEqual(len(selected), 2)
        self.assertIn("RSI_14", selected)
        self.assertIn("MACD", selected)
        self.assertNotIn("Poor_Feature", selected)
    
    def test_generate_evaluation_report(self):
        """测试评估报告生成"""
        # 创建模拟结果
        results = [
            FeatureEvaluationResult(
                feature_name="RSI_14",
                overall_score=0.85,
                individual_scores={'performance': 0.9, 'significance': 0.8},
                recommendation="ACCEPT - 强烈推荐使用",
                p_value=0.01,
                effect_size=0.5,
                confidence_interval=(0.1, 0.3),
                evaluation_details={}
            )
        ]
        
        report = self.evaluator.generate_evaluation_report(results)
        
        self.assertIsInstance(report, str)
        self.assertIn("特征评估报告", report)
        self.assertIn("RSI_14", report)
        self.assertIn("0.850", report)  # 评分
        self.assertIn("ACCEPT", report)  # 推荐
    
    def test_get_evaluation_summary(self):
        """测试评估摘要"""
        # 添加评估历史
        self.evaluator.evaluation_history = [
            FeatureEvaluationResult(
                feature_name="RSI_14",
                overall_score=0.85,
                individual_scores={},
                recommendation="ACCEPT - 强烈推荐使用",
                p_value=0.01,
                effect_size=0.5,
                confidence_interval=(0.1, 0.3),
                evaluation_details={}
            )
        ]
        self.evaluator.feature_rankings = {"RSI_14": 0.85}
        
        summary = self.evaluator.get_evaluation_summary()
        
        self.assertEqual(summary['total_evaluations'], 1)
        self.assertEqual(summary['mean_score'], 0.85)
        self.assertIn('recommendation_counts', summary)
        self.assertEqual(summary['recommendation_counts']['ACCEPT'], 1)
        self.assertIn("RSI_14", summary['top_features'])


class TestFeatureEvaluatorFactory(unittest.TestCase):
    """测试工厂函数"""
    
    def test_create_with_default_config(self):
        """测试使用默认配置创建"""
        evaluator = create_feature_evaluator()
        self.assertIsInstance(evaluator, FeatureEvaluator)
    
    def test_create_with_custom_config(self):
        """测试使用自定义配置创建"""
        config = Config()
        evaluator_config = FeatureEvaluatorConfig()
        
        evaluator = create_feature_evaluator(config, evaluator_config)
        self.assertIsInstance(evaluator, FeatureEvaluator)
        self.assertEqual(evaluator.config, config)
        self.assertEqual(evaluator.evaluator_config, evaluator_config)


class TestFeatureEvaluatorIntegration(unittest.TestCase):
    """集成测试"""
    
    def test_integration_with_features_module(self):
        """测试与特征工程模块的集成"""
        from src.features import FeatureEvaluator as ImportedEvaluator
        
        evaluator = ImportedEvaluator()
        self.assertIsInstance(evaluator, FeatureEvaluator)
    
    def test_integration_with_feature_engineer(self):
        """测试与FeatureEngineer的潜在集成"""
        from src.features import FeatureEngineer
        
        # 测试两个类可以在同一模块中共存
        engineer = FeatureEngineer()
        evaluator = FeatureEvaluator()
        
        self.assertIsInstance(engineer, FeatureEngineer)
        self.assertIsInstance(evaluator, FeatureEvaluator)


if __name__ == '__main__':
    unittest.main()