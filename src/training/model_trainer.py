"""
Model Trainer - Compatibility Wrapper
模型训练器 - 兼容性包装器

这是为了兼容性而提供的包装器，实际训练功能在stable_baselines_trainer.py中
"""

# 导入训练器
from .stable_baselines_trainer import StableBaselinesTrainer as ModelTrainer, TradingCallback
from .hyperparameter_optimizer import HyperparameterOptimizer

# 向后兼容性别名
TrainingCallback = TradingCallback

__all__ = ['ModelTrainer', 'TradingCallback', 'TrainingCallback', 'HyperparameterOptimizer']