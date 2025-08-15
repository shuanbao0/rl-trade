"""
Rewards Module - 独立的奖励函数模块

这个模块提供了完全独立的奖励函数系统，包括：
- 统一的奖励计算接口
- 智能的奖励函数工厂
- 插件化的扩展机制
- 市场类型和时间粒度的自适应配置

版本: 2.0.0
作者: TensorTrade Modern Team
"""

# 核心组件
from .core.reward_context import RewardContext, RewardResult, RewardContextBuilder
from .core.base_reward import BaseReward, RewardPlugin, SimpleRewardMixin, HistoryAwareRewardMixin
from .core.reward_registry import RewardRegistry, register_reward, get_global_registry
from .core.reward_factory import SmartRewardFactory, CompositeReward, create_reward, create_optimal_reward

# 版本信息
__version__ = "2.0.0"
__author__ = "TensorTrade Modern Team"

# 公共接口导出
__all__ = [
    # 核心类
    'RewardContext',
    'RewardResult', 
    'RewardContextBuilder',
    'BaseReward',
    'RewardPlugin',
    
    # 混入类
    'SimpleRewardMixin',
    'HistoryAwareRewardMixin',
    
    # 注册中心
    'RewardRegistry',
    'get_global_registry',
    'register_reward',
    
    # 工厂类
    'SmartRewardFactory',
    'CompositeReward',
    
    # 便捷函数
    'create_reward',
    'create_optimal_reward',
    
    # 向后兼容函数
    'create_reward_function',
    'get_reward_function_info',
]

# 向后兼容的便捷函数
def create_reward_function(reward_type: str = 'risk_adjusted', **kwargs) -> BaseReward:
    """
    向后兼容的奖励函数创建接口
    
    Args:
        reward_type: 奖励函数类型
        **kwargs: 配置参数
        
    Returns:
        BaseReward: 奖励函数实例
    """
    return create_reward(reward_type, **kwargs)


def get_reward_function_info() -> dict:
    """
    向后兼容的奖励函数信息获取接口
    
    Returns:
        dict: 奖励函数信息字典
    """
    registry = get_global_registry()
    info = {}
    
    for name, reward_class in registry.list_rewards().items():
        info[name] = {
            'class': reward_class.__name__,
            'description': getattr(reward_class, '__doc__', '无描述'),
            'aliases': getattr(reward_class, '_reward_aliases', [])
        }
    
    return info


# 模块级配置
_global_config = {
    'enable_performance_tracking': True,
    'enable_plugin_system': True,
    'default_cache_size': 128,
    'log_level': 'INFO'
}


def configure(config: dict):
    """
    配置奖励模块
    
    Args:
        config: 配置字典
    """
    global _global_config
    _global_config.update(config)
    
    # 使用统一日志系统
    from ..utils.logger import setup_logger
    setup_logger('rewards', level=_global_config['log_level'])


def get_config() -> dict:
    """获取当前配置"""
    return _global_config.copy()


# 模块初始化
def _initialize_module():
    """模块初始化"""
    # 使用统一日志系统
    from ..utils.logger import setup_logger, get_default_log_file
    
    # 设置奖励模块日志
    log_file = get_default_log_file('rewards')
    logger = setup_logger('rewards', level='INFO', log_file=log_file, console_output=True)
    
    logger.info(f"Rewards Module v{__version__} initialized")


# 执行初始化
_initialize_module()