"""
Smart Reward Factory - 智能奖励函数工厂
提供智能的奖励函数创建、配置和优化功能
"""

import logging
from typing import Dict, Any, Optional, Type, List
from .base_reward import BaseReward
from .reward_registry import RewardRegistry, get_global_registry
from .reward_context import RewardContext


class SmartRewardFactory:
    """
    智能奖励函数工厂
    
    功能：
    1. 基于上下文的智能参数调整
    2. 奖励函数实例缓存
    3. 性能优化配置
    4. 最佳匹配推荐
    """
    
    def __init__(self, registry: RewardRegistry = None, enable_cache: bool = True):
        """
        初始化智能工厂
        
        Args:
            registry: 奖励函数注册中心
            enable_cache: 是否启用实例缓存
        """
        self.registry = registry or get_global_registry()
        self.enable_cache = enable_cache
        self.cache: Dict[str, BaseReward] = {}
        
        # 配置适配器
        self._config_adapters: Dict[str, callable] = {}
        self._setup_default_adapters()
        
        # 使用统一日志系统
        from ...utils.logger import get_logger
        self.logger = get_logger("SmartRewardFactory")
        self.logger.info(f"智能奖励工厂初始化完成，缓存: {enable_cache}")
    
    def _setup_default_adapters(self):
        """设置默认的配置适配器"""
        
        # 时间粒度适配器
        self._config_adapters['granularity'] = self._adapt_for_granularity
        
        # 市场类型适配器
        self._config_adapters['market_type'] = self._adapt_for_market_type
        
        # 性能适配器
        self._config_adapters['performance'] = self._adapt_for_performance
    
    def create(self, 
               reward_name: str,
               context_hint: Optional[RewardContext] = None,
               force_new: bool = False,
               **config) -> BaseReward:
        """
        创建奖励函数实例
        
        Args:
            reward_name: 奖励函数名称
            context_hint: 上下文提示，用于智能配置调整
            force_new: 是否强制创建新实例
            **config: 配置参数
            
        Returns:
            BaseReward: 奖励函数实例
        """
        # 缓存检查
        cache_key = self._get_cache_key(reward_name, config)
        
        if (not force_new and self.enable_cache and 
            cache_key in self.cache):
            self.logger.debug(f"从缓存获取奖励函数: {reward_name}")
            return self.cache[cache_key]
        
        # 获取奖励函数类
        try:
            reward_class = self.registry.get(reward_name)
        except KeyError as e:
            # 尝试智能匹配
            suggestions = self._find_similar_rewards(reward_name)
            error_msg = f"奖励函数 '{reward_name}' 不存在"
            if suggestions:
                error_msg += f"。建议使用: {suggestions}"
            raise KeyError(error_msg) from e
        
        # 智能配置调整
        if context_hint:
            config = self._adjust_config_for_context(config, context_hint)
        
        # 应用配置适配器
        config = self._apply_config_adapters(reward_name, config)
        
        # 创建实例
        try:
            instance = reward_class(**config)
            
            # 缓存实例
            if self.enable_cache:
                self.cache[cache_key] = instance
            
            self.logger.info(f"创建奖励函数: {reward_name} with config: {config}")
            return instance
            
        except Exception as e:
            self.logger.error(f"创建奖励函数失败 {reward_name}: {e}")
            raise
    
    def create_optimal_for_market(self,
                                  market_type: str,
                                  granularity: str = None,
                                  strategy: str = "balanced",
                                  **config) -> BaseReward:
        """
        为特定市场创建最优奖励函数
        
        Args:
            market_type: 市场类型 (forex/stock/crypto)
            granularity: 时间粒度 (1min/5min/1h/1d)
            strategy: 策略类型 (conservative/balanced/aggressive)
            **config: 额外配置
            
        Returns:
            BaseReward: 最优奖励函数实例
        """
        # 获取适用的奖励函数列表
        candidates = self.registry.find_optimal(
            market_type=market_type,
            granularity=granularity
        )
        
        if not candidates:
            self.logger.warning(f"没有找到适用于 {market_type} 的奖励函数，使用默认")
            candidates = ['risk_adjusted']
        
        # 策略选择映射
        strategy_preferences = {
            'conservative': ['risk_adjusted', 'diversified', 'return_drawdown'],
            'balanced': ['forex_optimized', 'dynamic_sortino', 'regime_aware'] if market_type == 'forex'
                       else ['risk_adjusted', 'diversified', 'profit_loss'],
            'aggressive': ['curiosity_driven', 'self_rewarding', 'uncertainty_aware']
        }
        
        preferred = strategy_preferences.get(strategy, ['risk_adjusted'])
        
        # 选择最优奖励函数
        selected_reward = None
        for pref in preferred:
            if pref in candidates:
                selected_reward = pref
                break
        
        if selected_reward is None:
            selected_reward = candidates[0]
        
        # 创建实例，提供市场上下文
        return self.create(
            selected_reward,
            market_type=market_type,
            granularity=granularity,
            **config
        )
    
    def create_composite(self, 
                        reward_configs: List[Dict[str, Any]], 
                        weights: List[float] = None) -> BaseReward:
        """
        创建复合奖励函数
        
        Args:
            reward_configs: 奖励函数配置列表，每个包含 'name' 和其他参数
            weights: 权重列表，如未提供则均等权重
            
        Returns:
            BaseReward: 复合奖励函数
        """
        if not reward_configs:
            raise ValueError("至少需要一个奖励函数配置")
        
        # 创建子奖励函数
        sub_rewards = []
        for config in reward_configs:
            if 'name' not in config:
                raise ValueError("每个配置必须包含 'name' 字段")
            
            name = config.pop('name')
            reward = self.create(name, **config)
            sub_rewards.append(reward)
        
        # 设置权重
        if weights is None:
            weights = [1.0 / len(sub_rewards)] * len(sub_rewards)
        elif len(weights) != len(sub_rewards):
            raise ValueError("权重数量必须与奖励函数数量相匹配")
        
        # 创建复合奖励函数
        return CompositeReward(sub_rewards, weights)
    
    def _adjust_config_for_context(self, config: Dict[str, Any], context: RewardContext) -> Dict[str, Any]:
        """根据上下文调整配置参数"""
        adjusted = config.copy()
        
        # 根据时间粒度调整参数
        if context.granularity:
            granularity_adjustments = {
                '1min': {
                    'window_size': 20,
                    'target_scale': 0.1,
                    'risk_threshold': 0.005,
                    'transaction_cost_scale': 2.0
                },
                '5min': {
                    'window_size': 50,
                    'target_scale': 0.5,
                    'risk_threshold': 0.01,
                    'transaction_cost_scale': 1.0
                },
                '1h': {
                    'window_size': 24,
                    'target_scale': 2.0,
                    'risk_threshold': 0.02,
                    'transaction_cost_scale': 0.5
                },
                '1d': {
                    'window_size': 252,
                    'target_scale': 10.0,
                    'risk_threshold': 0.05,
                    'transaction_cost_scale': 0.1
                }
            }
            
            if context.granularity in granularity_adjustments:
                adjustment = granularity_adjustments[context.granularity]
                for key, value in adjustment.items():
                    if key not in adjusted:  # 只设置未明确指定的参数
                        adjusted[key] = value
        
        # 根据市场类型调整参数
        if context.market_type == 'forex':
            forex_adjustments = {
                'pip_size': 0.0001,
                'leverage': 100,
                'spread': 2,
                'daily_target_pips': 20
            }
            for key, value in forex_adjustments.items():
                if key not in adjusted:
                    adjusted[key] = value
        
        elif context.market_type == 'stock':
            stock_adjustments = {
                'commission': 0.001,
                'min_tick': 0.01,
                'market_hours': 6.5
            }
            for key, value in stock_adjustments.items():
                if key not in adjusted:
                    adjusted[key] = value
        
        elif context.market_type == 'crypto':
            crypto_adjustments = {
                'trading_hours': 24,
                'volatility_scale': 2.0,
                'min_order': 0.001
            }
            for key, value in crypto_adjustments.items():
                if key not in adjusted:
                    adjusted[key] = value
        
        return adjusted
    
    def _adapt_for_granularity(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """时间粒度适配器"""
        granularity = config.get('granularity')
        if not granularity:
            return config
        
        # 基于粒度的通用调整
        if '1min' in granularity:
            config['sensitivity'] = config.get('sensitivity', 1.5)
            config['noise_filter'] = config.get('noise_filter', True)
        elif '1d' in granularity:
            config['sensitivity'] = config.get('sensitivity', 0.5)
            config['trend_focus'] = config.get('trend_focus', True)
        
        return config
    
    def _adapt_for_market_type(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """市场类型适配器"""
        market_type = config.get('market_type')
        if not market_type:
            return config
        
        # 市场特定调整
        if market_type == 'forex':
            config['risk_free_rate'] = config.get('risk_free_rate', 0.02)
        elif market_type == 'stock':
            config['risk_free_rate'] = config.get('risk_free_rate', 0.03)
        elif market_type == 'crypto':
            config['volatility_adjustment'] = config.get('volatility_adjustment', 2.0)
        
        return config
    
    def _adapt_for_performance(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """性能适配器"""
        # 基于性能要求调整参数
        performance_mode = config.get('performance_mode', 'balanced')
        
        if performance_mode == 'fast':
            config['enable_caching'] = True
            config['simplified_calculation'] = True
        elif performance_mode == 'accurate':
            config['enable_caching'] = False
            config['precision_mode'] = True
        
        return config
    
    def _apply_config_adapters(self, reward_name: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """应用所有配置适配器"""
        for adapter_name, adapter_func in self._config_adapters.items():
            try:
                config = adapter_func(config)
            except Exception as e:
                self.logger.warning(f"配置适配器 {adapter_name} 失败: {e}")
        
        return config
    
    def _get_cache_key(self, reward_name: str, config: Dict[str, Any]) -> str:
        """生成缓存键"""
        # 对配置进行排序以确保一致的缓存键
        sorted_config = tuple(sorted(config.items()))
        return f"{reward_name}_{hash(sorted_config)}"
    
    def _find_similar_rewards(self, reward_name: str) -> List[str]:
        """查找相似的奖励函数名称"""
        from difflib import get_close_matches
        
        all_names = list(self.registry.list_all()) + list(self.registry.list_aliases().keys())
        matches = get_close_matches(reward_name, all_names, n=3, cutoff=0.6)
        
        return matches
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        return {
            'enabled': self.enable_cache,
            'size': len(self.cache),
            'keys': list(self.cache.keys())
        }
    
    def clear_cache(self):
        """清除缓存"""
        cleared_count = len(self.cache)
        self.cache.clear()
        self.logger.info(f"清除了 {cleared_count} 个缓存实例")
    
    def add_config_adapter(self, name: str, adapter_func: callable):
        """添加自定义配置适配器"""
        self._config_adapters[name] = adapter_func
        self.logger.info(f"添加配置适配器: {name}")


class CompositeReward(BaseReward):
    """复合奖励函数 - 组合多个奖励函数"""
    
    def __init__(self, rewards: List[BaseReward], weights: List[float]):
        """
        初始化复合奖励函数
        
        Args:
            rewards: 子奖励函数列表
            weights: 对应的权重列表
        """
        super().__init__(name="CompositeReward")
        self.rewards = rewards
        self.weights = weights
        
        if len(rewards) != len(weights):
            raise ValueError("奖励函数数量与权重数量不匹配")
        
        # 权重归一化
        weight_sum = sum(weights)
        if weight_sum > 0:
            self.weights = [w / weight_sum for w in weights]
    
    def calculate(self, context: RewardContext) -> float:
        """计算复合奖励"""
        total_reward = 0.0
        
        for reward_func, weight in zip(self.rewards, self.weights):
            try:
                sub_reward = reward_func(context)
                total_reward += sub_reward * weight
            except Exception as e:
                self.logger.warning(f"子奖励函数 {reward_func.name} 计算失败: {e}")
        
        return total_reward
    
    def get_info(self) -> Dict[str, Any]:
        """获取复合奖励函数信息"""
        sub_info = []
        for reward_func, weight in zip(self.rewards, self.weights):
            sub_info.append({
                'name': reward_func.name,
                'weight': weight,
                'info': reward_func.get_info()
            })
        
        return {
            'name': self.name,
            'type': 'composite',
            'sub_rewards': sub_info,
            'total_components': len(self.rewards)
        }
    
    def reset(self):
        """重置所有子奖励函数"""
        super().reset()
        for reward_func in self.rewards:
            reward_func.reset()


# 便捷函数
def create_reward(reward_name: str, **config) -> BaseReward:
    """
    便捷的奖励函数创建函数
    
    Args:
        reward_name: 奖励函数名称
        **config: 配置参数
        
    Returns:
        BaseReward: 奖励函数实例
    """
    factory = SmartRewardFactory()
    return factory.create(reward_name, **config)


def create_optimal_reward(market_type: str, granularity: str = None, 
                         strategy: str = "balanced", **config) -> BaseReward:
    """
    便捷的最优奖励函数创建函数
    
    Args:
        market_type: 市场类型
        granularity: 时间粒度
        strategy: 策略类型
        **config: 配置参数
        
    Returns:
        BaseReward: 最优奖励函数实例
    """
    factory = SmartRewardFactory()
    return factory.create_optimal_for_market(
        market_type=market_type,
        granularity=granularity,
        strategy=strategy,
        **config
    )