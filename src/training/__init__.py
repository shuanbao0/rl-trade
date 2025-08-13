"""
训练模块

基于Stable-Baselines3的强化学习训练系统，替代原有的Ray RLlib实现。
提供超参数优化、完整训练流水线和实验管理功能。

主要组件:
- StableBaselinesTrainer: 基于Stable-Baselines3的训练器
- HyperparameterOptimizer: 基于Optuna的超参数优化
- TrainingPipeline: 完整的训练流水线管理
"""

from .stable_baselines_trainer import StableBaselinesTrainer as ModernTrainer, quick_train
from .hyperparameter_optimizer import HyperparameterOptimizer, quick_optimize
from .training_pipeline import TrainingPipeline, run_quick_experiment

# 版本信息
__version__ = "1.0.0"

# 导出的主要类和函数
__all__ = [
    # 核心训练类
    'ModernTrainer',
    'StableBaselinesTrainer', 
    'HyperparameterOptimizer', 
    'TrainingPipeline',
    
    # 便捷函数
    'quick_train',
    'quick_optimize',
    'run_quick_experiment',
]

# 同时导出新名称以便直接使用
StableBaselinesTrainer = ModernTrainer

# 模块级别的文档字符串
def get_training_info():
    """
    获取训练模块信息
    
    Returns:
        Dict[str, Any]: 模块信息
    """
    return {
        'module': 'src.training',
        'version': __version__,
        'description': '强化学习训练系统',
        'framework': 'Stable-Baselines3',
        'optimizer': 'Optuna',
        'supported_algorithms': ['PPO', 'SAC', 'DQN'],
        'features': [
            '超参数自动优化',
            '完整训练流水线',
            '实验管理和结果追踪',
            '多种RL算法支持',
            '与奖励系统集成'
        ],
        'classes': {
            'StableBaselinesTrainer': '基于Stable-Baselines3的训练器',
            'HyperparameterOptimizer': '基于Optuna的超参数优化器',
            'TrainingPipeline': '完整的训练流水线管理器'
        }
    }


# 模块初始化日志
import logging
logger = logging.getLogger(__name__)
logger.info(f"训练模块加载完成 - 版本: {__version__}")
logger.info("支持的算法: PPO, SAC, DQN")
logger.info("集成组件: Stable-Baselines3 + Optuna")