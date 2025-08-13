"""
特征工程模块 - Enhanced with Experiment #005 Features
负责技术指标计算、统计特征生成、数据预处理等功能

新增 Experiment #005 特征：
- FeatureEvaluator: 科学的特征评估和选择框架
- 支持渐进式特征选择和统计显著性检验
"""

from .feature_engineer import FeatureEngineer
from .feature_evaluator import FeatureEvaluator, FeatureEvaluationResult, FeatureEvaluatorConfig, create_feature_evaluator

__all__ = [
    'FeatureEngineer', 
    'FeatureEvaluator',
    'FeatureEvaluationResult',
    'FeatureEvaluatorConfig', 
    'create_feature_evaluator'
] 