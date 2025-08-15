"""
Base Reward - 统一的奖励函数基类
完全独立于环境，提供标准化的奖励计算接口
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Callable
import time
import logging
import numpy as np
from .reward_context import RewardContext, RewardResult


class RewardPlugin(ABC):
    """奖励函数插件基类"""
    
    @abstractmethod
    def before_calculate(self, context: RewardContext) -> RewardContext:
        """奖励计算前的预处理"""
        pass
    
    @abstractmethod
    def after_calculate(self, context: RewardContext, reward: float) -> float:
        """奖励计算后的后处理"""
        pass
    
    def on_error(self, context: RewardContext, error: Exception):
        """错误处理"""
        pass
    
    def reset(self):
        """重置插件状态"""
        pass
    
    def attach(self, reward_function):
        """附加到奖励函数"""
        self.reward_function = reward_function


class BaseReward(ABC):
    """
    统一的奖励函数基类 - 完全独立于环境
    
    设计原则:
    1. 统一接口: 所有奖励函数使用相同的calculate()接口
    2. 上下文驱动: 通过RewardContext获取所需数据
    3. 插件支持: 支持插件扩展功能
    4. 性能跟踪: 内置性能监控和统计
    5. 错误处理: 完善的异常处理机制
    """
    
    def __init__(self, name: str = None, **config):
        """
        初始化奖励函数
        
        Args:
            name: 奖励函数名称
            **config: 配置参数
        """
        self.name = name or self.__class__.__name__
        self.config = config
        self.version = "2.0.0"
        
        # 性能跟踪
        self.call_count = 0
        self.total_compute_time = 0.0
        self.error_count = 0
        
        # 状态管理
        self.is_initialized = False
        self.last_context = None
        self.last_reward = None
        
        # 插件支持
        self.plugins: List[RewardPlugin] = []
        
        # 统一日志系统
        from ...utils.logger import get_logger
        self.logger = get_logger(f"Reward.{self.name}")
        
        # 内部状态
        self._internal_state = {}
        
        # 初始化
        self._initialize()
    
    def _initialize(self):
        """内部初始化"""
        try:
            self.setup()
            self.is_initialized = True
            self.logger.info(f"奖励函数 {self.name} 初始化成功")
        except Exception as e:
            self.logger.error(f"奖励函数 {self.name} 初始化失败: {e}")
            raise
    
    def setup(self):
        """子类可重写的初始化方法"""
        pass
    
    @abstractmethod
    def calculate(self, context: RewardContext) -> float:
        """
        核心奖励计算方法 - 必须由子类实现
        
        Args:
            context: 奖励计算上下文
            
        Returns:
            float: 奖励值
        """
        pass
    
    @abstractmethod
    def get_info(self) -> Dict[str, Any]:
        """
        获取奖励函数信息 - 必须由子类实现
        
        Returns:
            Dict: 包含奖励函数描述、参数、特性等信息
        """
        pass
    
    def reset(self):
        """重置奖励函数状态"""
        self.call_count = 0
        self.total_compute_time = 0.0
        self.error_count = 0
        self.last_context = None
        self.last_reward = None
        self._internal_state.clear()
        
        # 重置所有插件
        for plugin in self.plugins:
            plugin.reset()
        
        self.logger.debug(f"奖励函数 {self.name} 状态已重置")
    
    def add_plugin(self, plugin: RewardPlugin):
        """
        添加插件
        
        Args:
            plugin: 奖励函数插件
        """
        self.plugins.append(plugin)
        plugin.attach(self)
        self.logger.info(f"为 {self.name} 添加插件: {plugin.__class__.__name__}")
    
    def remove_plugin(self, plugin_class: type):
        """
        移除插件
        
        Args:
            plugin_class: 插件类
        """
        self.plugins = [p for p in self.plugins if not isinstance(p, plugin_class)]
        self.logger.info(f"从 {self.name} 移除插件: {plugin_class.__name__}")
    
    def __call__(self, context: RewardContext) -> float:
        """
        统一调用接口 - 提供插件支持和性能监控
        
        Args:
            context: 奖励计算上下文
            
        Returns:
            float: 最终奖励值
        """
        if not self.is_initialized:
            raise RuntimeError(f"奖励函数 {self.name} 未正确初始化")
        
        start_time = time.time()
        
        try:
            # 前置插件处理
            processed_context = context
            for plugin in self.plugins:
                try:
                    processed_context = plugin.before_calculate(processed_context)
                except Exception as e:
                    self.logger.warning(f"插件 {plugin.__class__.__name__} 前置处理失败: {e}")
            
            # 核心奖励计算
            reward = self.calculate(processed_context)
            
            # 验证奖励值
            reward = self._validate_reward(reward)
            
            # 后置插件处理
            final_reward = reward
            for plugin in self.plugins:
                try:
                    final_reward = plugin.after_calculate(processed_context, final_reward)
                except Exception as e:
                    self.logger.warning(f"插件 {plugin.__class__.__name__} 后置处理失败: {e}")
            
            # 更新统计信息
            self.call_count += 1
            self.total_compute_time += time.time() - start_time
            self.last_context = context
            self.last_reward = final_reward
            
            return final_reward
            
        except Exception as e:
            # 错误统计
            self.error_count += 1
            
            # 错误处理插件
            for plugin in self.plugins:
                try:
                    plugin.on_error(context, e)
                except Exception as plugin_error:
                    self.logger.error(f"插件错误处理失败: {plugin_error}")
            
            # 记录错误
            self.logger.error(f"奖励计算失败 {self.name}: {e}")
            
            # 根据配置决定是否抛出异常
            if self.config.get('raise_on_error', False):
                raise
            else:
                # 返回默认奖励值
                return self.config.get('default_reward', 0.0)
    
    def _validate_reward(self, reward: float) -> float:
        """
        验证和规范化奖励值
        
        Args:
            reward: 原始奖励值
            
        Returns:
            float: 验证后的奖励值
        """
        # 检查是否为有效数字
        if not isinstance(reward, (int, float)) or np.isnan(reward) or np.isinf(reward):
            self.logger.warning(f"无效的奖励值: {reward}, 使用默认值 0.0")
            return 0.0
        
        # 可选的奖励值范围限制
        if 'reward_range' in self.config:
            min_reward, max_reward = self.config['reward_range']
            reward = np.clip(reward, min_reward, max_reward)
        
        return float(reward)
    
    def calculate_with_result(self, context: RewardContext) -> RewardResult:
        """
        带详细结果的奖励计算
        
        Args:
            context: 奖励计算上下文
            
        Returns:
            RewardResult: 详细的奖励计算结果
        """
        start_time = time.time()
        
        try:
            reward = self(context)
            computation_time = time.time() - start_time
            
            return RewardResult(
                reward_value=reward,
                components=self._get_reward_components(context),
                metadata=self._get_computation_metadata(),
                computation_time=computation_time
            )
            
        except Exception as e:
            computation_time = time.time() - start_time
            
            return RewardResult(
                reward_value=0.0,
                components={},
                metadata={'error': str(e), 'error_type': type(e).__name__},
                computation_time=computation_time
            )
    
    def _get_reward_components(self, context: RewardContext) -> Dict[str, float]:
        """获取奖励组件分解 - 子类可重写"""
        return {'total': self.last_reward} if self.last_reward is not None else {}
    
    def _get_computation_metadata(self) -> Dict[str, Any]:
        """获取计算元数据 - 子类可重写"""
        return {
            'call_count': self.call_count,
            'avg_compute_time': self.get_avg_compute_time(),
            'error_rate': self.get_error_rate()
        }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计信息"""
        return {
            'name': self.name,
            'version': self.version,
            'call_count': self.call_count,
            'total_compute_time': self.total_compute_time,
            'avg_compute_time': self.get_avg_compute_time(),
            'error_count': self.error_count,
            'error_rate': self.get_error_rate(),
            'is_initialized': self.is_initialized,
            'plugins_count': len(self.plugins),
            'config': self.config
        }
    
    def get_avg_compute_time(self) -> float:
        """获取平均计算时间"""
        return self.total_compute_time / self.call_count if self.call_count > 0 else 0.0
    
    def get_error_rate(self) -> float:
        """获取错误率"""
        return self.error_count / self.call_count if self.call_count > 0 else 0.0
    
    def validate_context(self, context: RewardContext) -> bool:
        """
        验证上下文有效性 - 子类可重写
        
        Args:
            context: 奖励计算上下文
            
        Returns:
            bool: 是否有效
        """
        required_fields = ['portfolio_value', 'action', 'current_price', 'step']
        
        for field in required_fields:
            if not hasattr(context, field) or getattr(context, field) is None:
                self.logger.warning(f"上下文缺少必需字段: {field}")
                return False
        
        return True
    
    def __str__(self) -> str:
        """字符串表示"""
        return f"{self.name}(calls={self.call_count}, errors={self.error_count})"
    
    def __repr__(self) -> str:
        """详细表示"""
        return (f"{self.__class__.__name__}(name='{self.name}', "
                f"version='{self.version}', calls={self.call_count})")


class SimpleRewardMixin:
    """简单奖励函数混入 - 为简单奖励函数提供常用功能"""
    
    def get_step_return(self, context: RewardContext) -> float:
        """计算步骤收益率"""
        return context.get_step_return()
    
    def get_total_return(self, context: RewardContext) -> float:
        """计算总收益率"""
        return context.get_return_pct()


class HistoryAwareRewardMixin:
    """历史感知奖励函数混入 - 为需要历史数据的奖励函数提供工具"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.min_history_steps = kwargs.get('min_history_steps', 2)
    
    def has_sufficient_history(self, context: RewardContext) -> bool:
        """检查是否有足够的历史数据"""
        return context.has_sufficient_history(self.min_history_steps)
    
    def get_returns_series(self, context: RewardContext, window: int = None) -> np.ndarray:
        """获取收益率序列"""
        if context.portfolio_history is None or len(context.portfolio_history) < 2:
            return np.array([])
        
        values = context.portfolio_history
        if window:
            values = values[-window:]
        
        returns = np.diff(values) / values[:-1]
        return returns[~np.isnan(returns)]